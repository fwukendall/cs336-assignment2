uv run python infer.py --model_src ../data/results/full_v20260418_w1e3_s1e4_lr001-0001_b64_out.pkl --bpe_vocab_path ../data/results/train_bpe_tiny_train_vocab.pkl --bpe_merges_path ../data/results/train_bpe_tiny_train_merges.pkl --temp 0.5 --p 1.0 --seed 20260419 --max_out 256 --num_tries 3

uv run python infer.py --model_src ../data/results/full_v20260419_owt_w1e3_s4e4_lr001-0001_b64_checkpoint.pkl --bpe_vocab_path ../data/results/train_bpe_owt_train_vocab.pkl --bpe_merges_path ../data/results/train_bpe_owt_train_merges.pkl --temp 1.0 -p 0.999 --seed 20260420 --max_out 256 -num_tries 3
