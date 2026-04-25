import pickle
import copy
import os
from typing import BinaryIO
import regex as re
from typing import Tuple, Dict, List, Iterable, Iterator
from collections import Counter
from functools import partial
import multiprocessing as mp
import json
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_chunk_size: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    desired_num_chunks = file_size // desired_chunk_size + 1
    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    stlen = len(split_special_token)
    rewind_length = stlen - 1
    tail = b""

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = tail + file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
            tail = mini_chunk[-rewind_length:]

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(text: str, special_tokens=None) -> List[str]:
    out_pts = []
    if special_tokens:
        ssr = '|'.join([re.escape(st.decode('utf-8')) for st in special_tokens])
        last_end = 0
        for match in re.finditer(ssr, text):
            sub_text= text[last_end:match.start()]
            last_end = match.end()
            if not sub_text:
                out_pts.append(match.group())
                continue
            for pt in re.finditer(PAT, sub_text):
                out_pts.append(pt.group())
            out_pts.append(match.group())
        # take care of tail
        sub_text = text[last_end:]
        if sub_text:
            for pt in re.finditer(PAT, sub_text):
                out_pts.append(pt.group())
    else:
        for pt in re.finditer(PAT, text):
            out_pts.append(pt.group())
    return out_pts

def count_chunk_pretokens(chunk, pt_counts=None, special_tokens=None) -> Dict[Tuple, int]:
    if pt_counts is None:
        pt_counts = Counter()
    ssr = '|'.join([re.escape(st.decode('utf-8')) for st in special_tokens])
    last_end = 0
    assert special_tokens is not None
    for match in re.finditer(ssr, chunk):
        sub_chunk = chunk[last_end:match.start()]
        last_end = match.end()
        pt_counts[match.group()] += 1
        if not sub_chunk:
            continue
        for pt in re.finditer(PAT, sub_chunk):
            pt_counts[pt.group()] += 1
    # take care of tail
    sub_chunk = chunk[last_end:]
    if sub_chunk:
        for pt in re.finditer(PAT, sub_chunk):
            pt_counts[pt.group()] += 1
    return pt_counts

def mp_pt(chunk_start, chunk_end, input_path, special_tokens=None):
    with open(input_path, 'rb') as f:
        f.seek(chunk_start)
        chunk = f.read(chunk_end - chunk_start).decode("utf-8", errors="ignore")
    chunk_pt_count = count_chunk_pretokens(chunk, special_tokens=special_tokens)
    return chunk_pt_count

## Usage
def get_pretoken_counts(input_path, desired_chunk_size=int(1e6), num_chunks=None, num_processes=8,
                        special_tokens=None, special_split_token='<|endoftext|>'):
    pt_counts = None
    chunk_count = 0
    special_split_token = special_split_token.encode('utf-8')
    if special_tokens is None:
        special_tokens = [special_split_token]
    else:
        special_tokens = special_tokens + [special_split_token]
        special_tokens = sorted(set(special_tokens), key=lambda st: len(st), reverse=True)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_chunk_size, special_split_token)
    if num_processes == 1:
        # run serial
        with open(input_path, "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                pt_counts = count_chunk_pretokens(chunk, pt_counts=pt_counts, special_tokens=special_tokens)
                chunk_count += 1
                if num_chunks and chunk_count >= num_chunks:
                    break
    else:
        pt_call = partial(mp_pt, input_path=input_path, special_tokens=[special_split_token])
        n_proc = min(num_processes, len(boundaries) - 1)
        print(f'num boundaries {len(boundaries)}')
        pt_counts = Counter()
        if len(boundaries) >= 5 * n_proc:
            # very large file
            print(f'Very large files with {len(boundaries)}')
            stride = 5 * n_proc
            for base in range(0, len(boundaries), stride):
                with mp.Pool(n_proc) as pool:
                    # print(f'Spawning {n_proc} processes!')
                    b_s, b_e = max(0, base - 1), min(base + stride, len(boundaries))
                    chunk_pt_counts = pool.starmap(pt_call, zip(boundaries[b_s:b_e-1], boundaries[b_s+1:b_e]))
                pt_counts += sum(chunk_pt_counts, start=Counter())
        else:
            with mp.Pool(n_proc) as pool:
                # print(f'Spawning {n_proc} processes!')
                chunk_pt_counts = pool.starmap(pt_call, zip(boundaries[:-1], boundaries[1:]))
            pt_counts += sum(chunk_pt_counts, start=Counter())
    if len(pt_counts) > 2e6:
        print(f'Pre-filter num of distinct pre-token {len(pt_counts)}')
        pt_counts = {pt: num for pt, num in pt_counts.items() if num > 1}
        print(f'Post-filter num of distinct pre-token {len(pt_counts)}')
    return pt_counts

def count_subpts(substats, start_len=2, width=20, counter=None):
    if counter is None:
        counter = Counter()
    for pt, num_occur in substats.items():
        bstr = pt.encode('utf-8')
        for cur_len in range(start_len, start_len+width):
            if len(bstr) < cur_len:
                continue
            for start_pos in range(0, len(bstr)-cur_len+1):
                counter[tuple([bytes([c]) for c in bstr[start_pos:(start_pos+cur_len)]])] += num_occur
    return counter

def broadcast_merge(pt_split: List[bytes], merge: tuple[bytes, bytes], num_occur, counter=None):
    if counter is None:
        counter = Counter()
    m1, m2 = merge
    m3 = m1 + m2
    if m3 not in b''.join(pt_split) or m1 not in pt_split:
        return (pt_split, counter)
    if len(m3) >= 10:
        pairs = list(zip(pt_split[:-1], pt_split[1:]))
        if (m1, m2) not in pairs:
            return (pt_split, counter)
    can_find = True
    new_split = []
    sub_split = pt_split[:]
    while can_find:
        next_i = sub_split.index(m1)
        new_split += sub_split[:next_i]
        if len(sub_split) > next_i + 1 and sub_split[next_i+1] == m2:
            new_split.append(m3)
            sub_split = sub_split[next_i+2:]
        else:
            new_split.append(m1)
            sub_split = sub_split[next_i+1:]
        can_find = m1 in sub_split
    new_split += sub_split

    # if len(new_split) != len(pt_split):
    #     for i in range(len(new_split) - 1):
    #         counter[(new_split[i], new_split[i+1])] += num_occur

    #     for i in range(len(pt_split) - 1):
    #         counter[(pt_split[i], pt_split[i+1])] -= num_occur

    for i, g in enumerate(new_split):
        if g == m3:
            if i > 0 and new_split[i-1] != m3:
                counter[(new_split[i-1], m3)] += num_occur
                counter[(new_split[i-1], m1)] -= num_occur

            if i < len(new_split) - 1:
                counter[(m3, new_split[i+1])] += num_occur
                if new_split[i+1] != m3:
                    counter[(m2, new_split[i+1])] -= num_occur
                else:
                    counter[(m2, m1)] -= num_occur
    return new_split, counter

def pt_batch_merge(
    pt_stats: Dict[str, int],
    pt_splits: Dict[str, List[bytes]],
    str_lookup: Dict[str, bytes],
    merge: tuple[bytes, bytes],
) -> Tuple[Dict[str, List[bytes]], Dict[Tuple[bytes, bytes], int]]:
    change_counter = Counter()
    updated_splits = {}
    after_merge = b''.join(merge)
    for pt, num in pt_stats.items():
        if after_merge not in str_lookup[pt]:
            continue
        updated_splits[pt], change_counter = broadcast_merge(pt_splits[pt], merge, num, counter=change_counter)
    pt_splits.update(updated_splits)
    return pt_splits, change_counter

def safe_counter_add(d1: Counter, d2: Counter):
    for k, v in d2.items():
        if v > 0:
            d1[k] += v
        elif k in d1:
            d1[k] += v
    return

def safe_counter_deduct(d1: Counter, d2: Counter):
    for k, v in d2.items():
        if v < 0 and k in d1:
            d1[k] += v

class BytePairEncoder:
    def __init__(
        self,
        vocab: Dict[int, bytes] | None = None,
        merges: List[tuple[bytes, bytes]] | None = None,
        special_tokens: List[str] | None = None,
    ):
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = set()
        else:
            self.special_tokens = set([st.encode('utf-8') for st in special_tokens])

        # make sure vocab contains special_tokens
        if vocab is None:
            vocab = {}
        all_tokens = vocab.values()
        for st in self.special_tokens:
            if st not in all_tokens:
                vocab = copy.deepcopy(vocab)
        self.rocab = {token: i for i, token in vocab.items()}

        base = len(vocab)
        for st in self.special_tokens:
            if st not in all_tokens:
                vocab[base] = st
                self.rocab[st] = base
                base += 1
        self.vocab = vocab
        return
    
    def init_train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        pt_chunk_size=int(1e6),
        pt_num_chunks=None,
        pt_path=None,
        num_processes=16,
        special_split_token: str='<|endoftext|>',
    ):
        if special_split_token not in special_tokens:
            special_tokens.append(special_split_token)
        self.special_strings = special_tokens
        self.special_split_token = special_split_token

        special_tokens = [st.encode('utf-8') for st in special_tokens]
        special_tokens = sorted(set(special_tokens))

        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pt_chunk_size = pt_chunk_size
        self.pt_num_chunks = pt_num_chunks
        self.pt_path = pt_path
        self.num_processes = num_processes
        self.covered_len = 0
        self.trained_vocab = {}
        self.trained_rocab = {}
        self.trained_merges = []
        return
    
    def train(self):
        pt_stats = self.get_pt_stats()
        all_splits = {
            pt: [bytes([c]) for c in list(pt.encode('utf-8'))]
            for pt in pt_stats 
        }
        str_lookup = {
            pt: pt.encode('utf-8')
            for pt in pt_stats
        }

        vocab = {}
        rocab = {}
        merges = []
        base = len(vocab)
        for i in range(256):
            t = bytes([i])
            vocab[i] = t
            rocab[t] = i
        base = len(vocab)

        for i, st in enumerate(self.special_tokens):
            if st in rocab:
                continue
            vocab[base] = st
            rocab[st] = base
            base += 1
        base = len(vocab)
        
        num_remain = self.vocab_size - base
        
        sptc = count_subpts(pt_stats, 2, 1)
        # sort on count first descendingly, then on token-candidate lexigraphy ascendingly
        candidates = sorted(sptc.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)[:int(1.5*num_remain)+100]
        hot_sptc = Counter(dict(candidates))
        cold_sptc = Counter({k: v for k, v in sptc.items() if k not in hot_sptc})
        if cold_sptc:
            cold_max = max(cold_sptc.values())
        else:
            cold_max = 0
        # sptc = Counter(dict(candidates))

        checkpt = time.time()
        while base < self.vocab_size and candidates and candidates[0][1] > 0:
            new_merge, _ = candidates[0]
            _ = hot_sptc.pop(new_merge)

            merges.append(new_merge)
            new_token = new_merge[0] + new_merge[1]
            vocab[base] = new_token
            rocab[new_token] = base
            base += 1
            num_remain -= 1

            result = pt_batch_merge(pt_splits=all_splits, pt_stats=pt_stats, merge=new_merge, str_lookup=str_lookup)
            safe_counter_add(hot_sptc, result[1])
            safe_counter_deduct(cold_sptc, result[1])
            all_splits = result[0]

            candidates = sorted(hot_sptc.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)[:int(1.5*num_remain)+100]
            if cold_max >= candidates[0][1]:
                print('Fishing out of cold_sptc')
                cold_candidates = sorted(cold_sptc.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)
                for i in range(100):
                    if not cold_sptc:
                        break
                    hot_sptc[cold_candidates[i][0]] = cold_sptc.pop(cold_candidates[i][0])
                candidates = sorted(hot_sptc.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)[:int(1.5*num_remain)+100]
                if cold_sptc:
                    cold_max = max(cold_sptc.values())
                else:
                    cold_max = 0

            if len(hot_sptc) > 1.5 * len(candidates) + 500:
                print('Demoting some to cold_sptc')
                old_hot = hot_sptc
                hot_sptc = Counter(dict(candidates))
                cold_update = Counter({k: v for k, v in old_hot.items() if k not in hot_sptc})
                cold_sptc.update(cold_update)
                cold_max = max(cold_sptc.values())

                for i in range(6):
                    if len(cold_sptc) > 5 * self.vocab_size:
                        print('Deleting some pairs from cold_sptc')
                        cold_sptc = Counter({k: v for k, v in cold_sptc.items() if v >= i * 0.1 * cold_max})
            
            if base % 500 == 0:
                new_checkpt = time.time()
                time_taken = int(new_checkpt - checkpt)
                print(f'Onto token number {base} time spent {time_taken}s')
                checkpt = new_checkpt

        self.sptc = sptc
        self.candidates = candidates
        self.trained_vocab = vocab
        self.trained_rocab = rocab
        self.trained_merges = merges
        return

    def get_pt_stats(self):
        input_path = self.input_path
        vocab_size = self.vocab_size
        special_tokens = self.special_tokens
        pt_chunk_size = self.pt_chunk_size
        pt_num_chunks = self.pt_num_chunks
        pt_path = self.pt_path
        num_processes = self.num_processes
        special_split_token = self.special_split_token

        if pt_path and os.path.exists(pt_path):
            print('Loading dumped pre-token counts!')
            with open(pt_path, 'r') as pf:
                pt_counts = json.load(pf)
        else:
            pt_counts = get_pretoken_counts(
                input_path=input_path, 
                desired_chunk_size=pt_chunk_size,
                num_chunks=pt_num_chunks,
                num_processes=num_processes,
                special_tokens=special_tokens,
                special_split_token=special_split_token,
            )
            print(f'Pre-tokenization finished with {len(pt_counts)} pre-tokens')
            if pt_path:
                with open(pt_path, 'w') as pf:
                    json.dump(pt_counts, pf)
                print('Written pre-token counts to', pt_path)
        
        # pre-processing
        for special_str in self.special_strings:
            if special_str in pt_counts:
                _ = pt_counts.pop(special_str)
        return pt_counts

    def encode(self, text:str) -> List[int]:
        pre_tokens = pre_tokenize(text, special_tokens=self.special_tokens)
        unique_pts = set(pre_tokens)
        transformed_pts = {}
        merges = self.merges
        for pt in unique_pts:
            pt_bstr = pt.encode('utf-8')
            if pt_bstr in self.special_tokens:
                # skip merging, special token!
                transformed_pts[pt] = [self.rocab[pt_bstr]]
                continue
            # apply merges
            tlist = [bytes([c]) for c in pt_bstr]
            pairs = list(zip(tlist[:-1], tlist[1:]))
            for merge in merges:
                if merge not in pairs:
                    continue
                
                m1, m2 = merge
                mr = b''.join(merge)
                can_find = True
                new_split = []
                while can_find:
                    next_i = tlist.index(m1)
                    new_split += tlist[:next_i]
                    if len(tlist) > next_i + 1 and tlist[next_i+1] == m2:
                        new_split.append(mr)
                        tlist = tlist[next_i+2:]
                    else:
                        new_split.append(m1)
                        tlist = tlist[next_i+1:]
                    can_find = m1 in tlist
                new_split += tlist
                tlist = new_split
                pairs = list(zip(tlist[:-1], tlist[1:]))
            transformed_pts[pt] = [self.rocab[t] for t in tlist]
        # we can actually know the size of this tokens list
        # check back to see if this needs to be allocated as a fixed size arr
        tokens = []
        for pt in pre_tokens:
            tokens.extend(transformed_pts[pt])
        return tokens

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        cur_chunk = ''
        cum_size = 0
        ssr = '|'.join([re.escape(st.decode('utf-8')) for st in self.special_tokens])
        max_token_len = max([len(t) for t in self.rocab])
        if self.special_tokens:
            max_st_len = max([len(st) for st in self.special_tokens])
        else:
            max_st_len = 0
        for i in range(12, 16):
            if 2 ** i > 100 * max_token_len:
                break
        chunk_size = 2 ** i
        rewind_len = max_st_len - 1
        assert chunk_size > max_st_len

        for text in iterable:
            exhausted = False
            new_size = len(text)
            new_base = 0
            while not exhausted:
                if cum_size + new_size > chunk_size:
                    proposed_cut = new_base + chunk_size - cum_size
                    new_chunk = cur_chunk + text[new_base:proposed_cut]
                    last_end = 0
                    if ssr != '':
                        for match in re.finditer(ssr, new_chunk):
                            sub_text = new_chunk[last_end:match.start()]
                            last_end = match.end()
                            tokens = self.encode(sub_text)
                            for token in tokens:
                                yield token
                            yield self.encode(match.group())
                    sub_text = new_chunk[last_end:]
                    if not sub_text:
                        cur_chunk = ''
                        cum_size = 0
                    else:
                        tokens = self.encode(sub_text[:-rewind_len])
                        cur_chunk = sub_text[-rewind_len:]
                        cum_size = rewind_len
                        for token in tokens:
                            yield token
                    new_base = proposed_cut
                    new_size = len(text) - new_base
                else:
                    cur_chunk += text
                    exhausted = True
        tokens = self.encode(cur_chunk)
        for token in tokens:
            yield token
        return
    
    def decode(self, ids: List[int]) -> str:
        stride = 1000
        out_bytes = b''

        for i in range(0, len(ids), stride):
            start, end = i, min(i + stride, len(ids))
            tokens = [self.vocab[id] for id in ids[start:end]]
            out_bytes += b''.join(tokens)
        
        out_string = out_bytes.decode('utf-8', errors='replace')
        return out_string

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, 'rb') as mf:
            merges = pickle.load(mf)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    pt_chunk_size=int(1e6),
    pt_num_chunks=None,
    pt_path=None,
    num_processes=16,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    bpencoder = BytePairEncoder()
    bpencoder.init_train(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        pt_chunk_size=pt_chunk_size,
        pt_num_chunks=pt_num_chunks,
        pt_path=pt_path,
        num_processes=num_processes,
        special_split_token='<|endoftext|>',
    )
    bpencoder.train()
    return (bpencoder.trained_vocab, bpencoder.trained_merges)

if __name__ == '__main__':
    test_path = '../tests/fixtures/corpus.en'
    owt_train_path = '../data/owt_train.txt'
    owt_valid_path = '../data/owt_valid.txt'
    tiny_train_path = '../data/TinyStoriesV2-GPT4-train.txt'
    bpeobj = BytePairEncoder()
    bpeobj.init_train(
        # input_path=owt_train_path,
        input_path=tiny_train_path,
        vocab_size=10000,
        special_tokens=['<|endoftext|>'],
        pt_chunk_size=int(4e6),
        # pt_num_chunks=50,
        # pt_path='../data/owt_train_pt_counts.json',
        # pt_path='../data/tinystories_train_pt_counts.json',
        num_processes=1,
    )
    # bpeobj.get_pt_stats()
    bpeobj.train()
    # out_vocab_path = '../data/results/train_bpe_owt_train_vocab.pkl'
    # out_merges_path = '../data/results/train_bpe_owt_train_merges.pkl'
    # with open(out_vocab_path, 'wb') as of:
    #     pickle.dump(bpeobj.trained_vocab, of)

    # with open(out_merges_path, 'wb') as of:
    #     pickle.dump(bpeobj.trained_merges, of)
    # print(f'Written {out_vocab_path}')
    # print(f'Written {out_merges_path}')
