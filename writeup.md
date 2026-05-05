### TODOs 20260501
- Single GPU A-100:
    - with autocast bf16, run benchmarking script of all model sizes, context-length 512 (small, medium, large, XL)
    - Run memory profiling for XL-128 and XL-512
    - Run memory profiling for compiled-checkpointed XL-2048, and compare XL-512
    - Run benchmark-attn for bmine attn impl (compile, no-compile)


## Problem (benchmarking_script):  Benchmarking Script (4 points)
**(a) Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass, and optimizer step in your model. Specifically, your script should support the following:**
• Given hyperparameters (e.g., number of layers), initialize a model.
• Generate a random batch of data.
• Run 𝑤 warm-up steps (before you start measuring time), then time the execution of 𝑛 steps (either only forward, forward and backward, or forward and backward with optimizer step, depending on an argument). For timing, you can use the Python timeit module (e.g., either using the timeit function, or using timeit.default_timer(), which gives you the system’s highest resolution clock, thus a better default for benchmarking than time.time()).
• Call torch.cuda.synchronize() after each step.

Deliverable: A script that will initialize a basics Transformer model with the given hyperparameters, create a random batch of data, and time forward-only, forward-andbackward, and full training steps that include the optimizer step.

See `cs336_systems/benchmark.py`

**(b) Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?**

Deliverable: A 1-2 sentence response with your timings.

|             |       fw |    fw-bw |   fw-bw-opt |
|:------------|---------:|---------:|------------:|
| small_mean  | 0.046714 | 0.139415 |    0.145516 |
| small_std   | 0.000205 | 0.000377 |    5.6e-05  |
| medium_mean | 0.14021  | 0.415941 |    0.438516 |
| medium_std  | 0.002314 | 0.000698 |    0.000222 |
| large_mean  | 0.284852 | 0.884038 |    0.944976 |
| large_std   | 0.000472 | 0.000936 |    0.001349 |
| xl_mean     | 0.842428 | 2.56728  |    2.76223  |
| xl_std      | 0.000107 | 0.000688 |    0.000935 |

For the largest model (XL) I can fit on my A100 pod, fw and bw take 0.84 and 1.72 seconds respectively. Optimization step comes pretty cheap at a 0.2 seconds extra.

Standard deviation is very small.

**(c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?**

Deliverable: A 2-3 sentence response.

warmup 0

|             |       fw |    fw-bw |   fw-bw-opt |
|:------------|---------:|---------:|------------:|
| small_mean  | 0.06788  | 0.160079 |    0.160505 |
| small_std   | 0.061584 | 0.056885 |    0.041367 |
| medium_mean | 0.141674 | 0.419132 |    0.452641 |
| medium_std  | 0.007356 | 0.009058 |    0.042201 |
| large_mean  | 0.288382 | 0.888093 |    0.967655 |
| large_std   | 0.010646 | 0.009861 |    0.070559 |
| xl_mean     | 0.847424 | 2.57277  |    2.766    |
| xl_std      | 0.014875 | 0.017565 |    0.013652 |

warmup 1

|             |       fw |    fw-bw |   fw-bw-opt |
|:------------|---------:|---------:|------------:|
| small_mean  | 0.048888 | 0.141979 |    0.147706 |
| small_std   | 0.00384  | 0.004931 |    0.004593 |
| medium_mean | 0.139657 | 0.415778 |    0.438866 |
| medium_std  | 0.005339 | 0.000583 |    0.000708 |
| large_mean  | 0.283967 | 0.885075 |    0.944251 |
| large_std   | 0.00013  | 0.00208  |    0.001142 |
| xl_mean     | 0.842227 | 2.56875  |    2.76228  |
| xl_std      | 0.000166 | 0.001822 |    0.000781 |


## Problem (nsys_profile):  Nsight Systems Profiling (5 points)
**Profile your forward pass, backward pass, and optimizer step using nsys with two model sizes from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where the largest available size should be the longest context length you can fit in memory. Pick the combinations you think would be the most interesting to look at. For each profile answer the following questions:**

**(a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?**

Deliverable: A 1-2 sentence response.


I ran medium and large with 256, 512 and 1024. Time are in milliseconds.

Comparing against the benchmark table (CL-512 only), the times match up pretty well (0.132s v.s. 0.139s for medium, 0.277s v.s. 0.284s for large).

| model   |   cl |   bmmode_bw |   bmmode_fw |   bmmode_fw_withgrad |   bmmode_opt |
|:--------|-----:|------------:|------------:|---------------------:|-------------:|
| large   |  256 |     276.373 |    134.06   |              148.301 |      64.5769 |
| large   |  512 |     585.362 |    276.942  |              302.985 |      64.5771 |
| large   | 1024 |    1335.43  |    626.814  |              681.341 |      64.5124 |
| medium  |  256 |     126.771 |     63.0879 |               71.775 |      24.271  |
| medium  |  512 |     269.97  |    132.119  |              148.863 |      24.0014 |
| medium  | 1024 |     615.493 |    295.082  |              330.873 |      24.2909 |


**(b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the “CUDA GPU Kernel Summary” under “Stats System View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)**

Deliverable: A 1-2 sentence response.

In forward pass, the kernel that takes up the most GPU time is sgemm (matrix multiplication). Yes it's the most time-consuming kernel in both forward and backward. In the following table I group-summed all sgemm (of different tile-sizes).


| model   |   cl | bmmode             | kernel                 |   runtime_ms |   invokes |
|:--------|-----:|:-------------------|:-----------------------|-------------:|----------:|
| large   |  256 | bmmode_bw          | ampere_sgemm           |     234.913  |       506 |
| large   |  256 | bmmode_fw          | ampere_sgemm           |     113.845  |       254 |
| large   |  256 | bmmode_fw_withgrad | ampere_sgemm           |     122.335  |       274 |
| large   |  256 | bmmode_opt         | vectorized_elementwise |      64.5769 |      2805 |
| large   |  512 | bmmode_bw          | ampere_sgemm           |     448.582  |       434 |
| large   |  512 | bmmode_fw          | ampere_sgemm           |     229.106  |       254 |
| large   |  512 | bmmode_fw_withgrad | ampere_sgemm           |     245.487  |       273 |
| large   |  512 | bmmode_opt         | vectorized_elementwise |      64.5771 |      2805 |
| large   | 1024 | bmmode_bw          | ampere_sgemm           |     782.971  |       397 |
| large   | 1024 | bmmode_fw          | ampere_sgemm           |     488.039  |       254 |
| large   | 1024 | bmmode_fw_withgrad | ampere_sgemm           |     516.284  |       272 |
| large   | 1024 | bmmode_opt         | vectorized_elementwise |      64.5124 |      2805 |
| medium  |  256 | bmmode_bw          | ampere_sgemm           |     101.107  |       314 |
| medium  |  256 | bmmode_fw          | ampere_sgemm           |      51.8308 |       170 |
| medium  |  256 | bmmode_fw_withgrad | ampere_sgemm           |      57.0349 |       188 |
| medium  |  256 | bmmode_opt         | vectorized_elementwise |      24.271  |      1881 |
| medium  |  512 | bmmode_bw          | ampere_sgemm           |     207.922  |       314 |
| medium  |  512 | bmmode_fw          | ampere_sgemm           |     105.595  |       170 |
| medium  |  512 | bmmode_fw_withgrad | ampere_sgemm           |     116.624  |       190 |
| medium  |  512 | bmmode_opt         | vectorized_elementwise |      24.0014 |      1881 |
| medium  | 1024 | bmmode_bw          | ampere_sgemm           |     453.982  |       338 |
| medium  | 1024 | bmmode_fw          | ampere_sgemm           |     220.532  |       171 |
| medium  | 1024 | bmmode_fw_withgrad | ampere_sgemm           |     244.754  |       191 |
| medium  | 1024 | bmmode_opt         | vectorized_elementwise |      24.2909 |      1881 |

**(c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?**

Deliverable: A 1-2 sentence response.

Elementwise and vectorized_elementwise together accounts for 11-18% of forward runtime, with bigger runtime pct on larger context-length.

**(d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?**
Deliverable: A 1-2 sentence response.

For total time, as well as top kernel, see my answer to (a) and (b). As for top kernel time percentage, see following table. For foward and backward, matmul kernel runtime percentage drop as context-length gets bigger (holding `d_model` constant), but increases as the `d_model` gets bigger (holding `context_length` constant). Optimizer step only invokes the vectorized_elementwise kernel.


| model   |   cl | bmmode             | kernel                 |   time_pct |
|:--------|-----:|:-------------------|:-----------------------|-----------:|
| large   |  256 | bmmode_bw          | ampere_sgemm           |    84.9985 |
| large   |  256 | bmmode_fw          | ampere_sgemm           |    84.9207 |
| large   |  256 | bmmode_fw_withgrad | ampere_sgemm           |    82.4911 |
| large   |  256 | bmmode_opt         | vectorized_elementwise |   100      |
| large   |  512 | bmmode_bw          | ampere_sgemm           |    76.6334 |
| large   |  512 | bmmode_fw          | ampere_sgemm           |    82.727  |
| large   |  512 | bmmode_fw_withgrad | ampere_sgemm           |    81.0228 |
| large   |  512 | bmmode_opt         | vectorized_elementwise |   100      |
| large   | 1024 | bmmode_bw          | ampere_sgemm           |    58.6305 |
| large   | 1024 | bmmode_fw          | ampere_sgemm           |    77.8603 |
| large   | 1024 | bmmode_fw_withgrad | ampere_sgemm           |    75.7746 |
| large   | 1024 | bmmode_opt         | vectorized_elementwise |   100      |
| medium  |  256 | bmmode_bw          | ampere_sgemm           |    79.7556 |
| medium  |  256 | bmmode_fw          | ampere_sgemm           |    82.1566 |
| medium  |  256 | bmmode_fw_withgrad | ampere_sgemm           |    79.4634 |
| medium  |  256 | bmmode_opt         | vectorized_elementwise |   100      |
| medium  |  512 | bmmode_bw          | ampere_sgemm           |    77.0168 |
| medium  |  512 | bmmode_fw          | ampere_sgemm           |    79.9242 |
| medium  |  512 | bmmode_fw_withgrad | ampere_sgemm           |    78.3428 |
| medium  |  512 | bmmode_opt         | vectorized_elementwise |   100      |
| medium  | 1024 | bmmode_bw          | ampere_sgemm           |    73.7591 |
| medium  | 1024 | bmmode_fw          | ampere_sgemm           |    74.7358 |
| medium  | 1024 | bmmode_fw_withgrad | ampere_sgemm           |    73.9723 |
| medium  | 1024 | bmmode_opt         | vectorized_elementwise |   100      |

**(e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?**

Deliverable: A 1-2 sentence response.

For small model, breakdown is as follow

|         |   runtime_ms |   FLOPs |
|:--------|-------------:|--------:|
| QK      |        3.117 |     403 |
| softmax |        3.909 |       9 |
| smV     |        2.83  |     403 |

Note that softmax has 2% of the FLOPs of the matmuls, yet 30% longer in runtime.


## Problem (mixed_precision_accumulation):  Mixed-Precision Accumulation (1 point)

**Run the following code and comment on the accuracy of the results.**

```python
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(s)
 
s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)
 
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)
 
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
```

- (1) all-fp32: highest accuracy, error 1e-4
- (2) fp16 init, fp16 iter: lowest accuracy, error 5e-2
- (3) fp32 init, fp16 iter: medium accuracy, error 2e-3, vastly lower than (2)
- (4) fp32 init, fp16 iter casted to fp32: exactly the same as (3)

## Problem (benchmarking_mixed_precision): 2 points
**(a) Consider the following model:**

```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
```

**Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:**

• the model parameters within the autocast context,
• the output of the first feed-forward layer (ToyModel.fc1),
• the output of layer norm (ToyModel.ln),
• the model’s predicted logits,
• the loss,
• and the model’s gradients?

Deliverable: The data types for each of the components listed above.

- all model parameters remain fp32
- output of fc1 is fp16
- output of layernorm is fp32
- output of fc2 (predicted logits) is fp16
- loss is fp32
- all gradients are fp32


**(b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently?  Why or why not?**

Deliverable: A 2-3 sentence response.

With bfloat16, the layernorm output still remain in fp32. All my answers to the previous question remain true except to replace fp16 with bf16. Since the layernorm division can be quite precision-sensitive, especially with the denom and numer being off by multiple orders of magnitudes, it seems important for it to remain in higher precision.

**(c) Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in §1.1.2. Compare the results of using full vs. mixed precision, and comment on any trends as model size changes. You may find the nullcontext no-op context manager to be useful.**

Deliverable: A 2-3 sentence response with your timings and commentary.

TODO


## Problem (memory_profiling):  Memory Profiling (4 points)
Profile your complete training step of forward pass, backward pass, and optimizer step of the xl model from Table 1 with context lengths of 128 and 512.
**(a) Add an option to your profiling script to run your model through the memory profiler.  It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the xl model when either doing inference only (just forward pass) or a full training step. What do your memory timelines look like? Can you tell which stage is running based on the peaks you see?**

Deliverable: Two images of the “Active memory timeline” of an xl model, from the memory_viz tool: one for the forward pass, and one for running a full training step (forward and backward passes, then optimizer step), and a 2-3 sentence response.

TODO


**(b) What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?**

Deliverable: A table with two numbers per context length.

**(c) Find the peak memory usage of the xl model when using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage?**

Deliverable: A 2-3 sentence response.

TODO

**(d) Consider the xl model. Given our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MiB (i.e., divide the number of bytes by 1024^2).**

Deliverable: A 1-2 sentence response with your derivation.

```
size_128 = 4 * 4 * 128 * 2560 = 5 MiB
size_512 = 4 * 4 * 512 * 2560 = 20 MiB
size_2048 = 4 * 4 * 2048 * 2560 = 80 MiB
```

**(e) Now look closely at the “Active Memory Timeline” from pytorch.org/memory_viz of a memory snapshot of the xl model doing a forward pass. When you reduce the “Detail” level, the tool hides the smallest allocations to the corresponding level (e.g., putting “Detail” at 10% only shows the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?**

Deliverable: A 1-2 sentence response.

TODO

**(f) Nsight Systems also has flags for memory profiling. You can combine these with the Nsight flags from before to understand what allocations are happening at different steps in your model’s lifespan. Use the PyTorch-provided NVTX labels to determine how much memory is saved for backward (these tensors are often called residuals) by a single TransformerBlock in your model. Note the 5 largest contributing operations, and what percentage of the overall memory they contribute.  During the backward pass, all these tensors will be freed, but new gradient tensors are emitted at the same time. Based on your profiles showing how much memory was allocated during the forward pass, and how much memory usage changes for every TransformerBlock in the backward pass, calculate how much memory the produced gradient tensors for a TransformerBlock take. Does the result match what you expect?**

Deliverable: Screenshots from Nsight Systems and a 1-2 paragraph response.

TODO

## Problem (gradient_checkpointing):  Memory-Optimal Gradient Checkpointing (4 points)

**Consider a Transformer with 𝑁 identical blocks stacked sequentially. Without any checkpointing, all 𝑁 blocks’ worth of residuals are kept alive simultaneously, giving 𝑂(𝑁) peak activation memory. We have a free hand to wrap any subset of the forward pass in checkpoint, including nesting checkpoint calls inside one another.**

**(a) What checkpointing strategy minimizes peak activation memory, ignoring the compute cost?  Describe how you would arrange the checkpoint calls (a code sketch is fine), and give the asymptotic peak activation memory and compute of your strategy as a function of 𝑁. Assume the residuals saved by a single block dominate any per-checkpoint bookkeeping.**

Deliverable: A 3-5 sentence description of the strategy and its asymptotic peak memory, plus a short code sketch.

If I ignore compute cost, I would checkpoint only the initial input and
recursively re-compute all following blocks when doing backward.

```python
from torch.utils.checkpoint import checkpoint
from functools
# assume the block forward is block(x)
def o1_checkpoint(x, n: int):
    if n == 1:
        return block(x)
    f = partial(o1_checkpoint, n=n-1)
    x = checkpoint(f, x, use_reentrant=False)
    return block(x)
```

**(b) Consider the xl model config with batch size 4 and sequence length 2048 as above. If you only have the time/compute budget to run one step of recomputation (meaning you may not nest checkpoint calls), what is the best checkpointing strategy to reduce peak memory? Profile your run’s peak memory to validate your hypothesis. Compare the peak memory of the next smaller and larger checkpointing block sizes to be sure.**

Deliverable: A 3-5 sentence description of your reasoning along with the measured peak memory for your strategy.

Assuming we'll not checkpoint *within* a transformer block, the optimal strategy should've been to checkpoint the residual to every transformer block. Theoretically, if the blocks are infinitely slice-able, the optimal would be to stop at the point where introducing 1 more checkpoint does not decrease peak activation mem. Assuming:
G
```
N blocks (but infinitely separable)
residual tensor mem footprint is 1
each block's activation footprint is m
```

Then the optimal number of checkpoint is
```
argmin_K(K + mN/K), minimized at K = sqrt(m*N)
```

Without flash attention (which I understand takes out the `d_model ** 2 * c_l` and `c_l ** 2 * d_model` mem blackholes), m is larger than N for XL model (m~=70 while N=32), that the optimal is just to slice into N chunks of 1 transformer blocks.

Smoke run on `small` config with CL-512, BF16 mixed precision:
- Without any optimization, mem usage on fw-bw peaks at 3.7GiB
- With compile on each transformer layer, mem usage peaks at 2.7GiB
- With compile *and* per-layer checkpointing, mem usage peaks at 1.6GiB

Theoretically "raw" activation mem usage should be 2.4GiB, and with full checkpointing it should be 175MB on activation only. So in theory checkpointing should save 2.2GiB, against a measured reduction of 2.1GiB


With a checkpoint inserted at any

```
# no compile, no checkpointing 
# CL 256
                  fw     fw-bw  fw-bw-opt
small_mean  0.223222  0.742219   0.880503
small_std   0.002731  0.008660   0.026013

# CL 512
                  fw     fw-bw  fw-bw-opt
small_mean  0.519028  2.290702   4.958452
small_std   0.005325  0.275684   0.099407

# compile, no checkpointing
# CL 256
                  fw     fw-bw  fw-bw-opt
small_mean  0.154870  0.532780   0.710786
small_std   0.000463  0.018181   0.024915

# CL 512
                  fw     fw-bw  fw-bw-opt
small_mean  0.324337  0.997056   1.329358
small_std   0.000402  0.031531   0.054117

# compile, checkpointing
# CL 256
                  fw     fw-bw  fw-bw-opt
small_mean  0.159511  0.651433   0.814780
small_std   0.000147  0.003435   0.001873

# CL 512
                  fw     fw-bw  fw-bw-opt
small_mean  0.306798  1.226523   1.391464
small_std   0.001718  0.000981   0.001952
```

## Problem (pytorch_attention):  PyTorch Attention Benchmarking (2 points)
**(a) Benchmark your attention implementation at different scales. Write a script that will:**
- (i) Fix the batch size to 8 and don’t use multihead attention (i.e. remove the head dimension).
- (ii) Iterate through the cartesian product of [16, 32, 64, 128] for the head embedding dimension 𝑑model, and [256, 1024, 4096, 8192, 16384] for the sequence length.
- (iii) Create random inputs 𝑄, 𝐾, 𝑉 for the appropriate size.
- (iv) Time 100 forward passes through attention using the inputs.
- (v) Measure how much memory is in use before the backward pass starts, and time 100 backward passes.
- (vi) Make sure to warm up, and to call torch.cuda.synchronize() after each forward/backward pass.

**Depending on your GPU, some of these configurations are expected to run out of memory.  Report the timings (or out-of-memory errors) you get for these configurations. At what size do you get out-of-memory errors? Do the accounting for the memory usage of attention in one of the smallest configurations you find that runs out of memory (you can use the equations for memory usage of Transformers from Assignment 1). How does the memory saved for backward change with the sequence length? What would you do to eliminate this memory cost?**

Deliverable: A table with your timings, your calculations for the memory usage, and a 1-2 paragraph response.

| d_k,seq_len   |   afunc-bmine_nocomp_bf16_fw |   afunc-bmine_nocomp_bf16_bw |
|:--------------|-----------------------------:|-----------------------------:|
| (16, 256)     |                  0.000941632 |                   0.00173682 |
| (16, 1024)    |                  0.00965486  |                   0.0175487  |
| (16, 4096)    |                  0.0678801   |                   0.147964   |
| (16, 8192)    |                nan           |                 nan          |
| (16, 16384)   |                nan           |                 nan          |
| (32, 256)     |                  0.00117213  |                   0.00146997 |
| (32, 1024)    |                  0.00366305  |                   0.00719732 |
| (32, 4096)    |                  0.198743    |                   0.755467   |
| (32, 8192)    |                nan           |                 nan          |
| (32, 16384)   |                nan           |                 nan          |
| (64, 256)     |                  0.000777457 |                   0.0012469  |
| (64, 1024)    |                  0.00349827  |                   0.00712543 |
| (64, 4096)    |                  0.198169    |                   0.74446    |
| (64, 8192)    |                nan           |                 nan          |
| (64, 16384)   |                nan           |                 nan          |
| (128, 256)    |                  0.00059489  |                   0.00123288 |
| (128, 1024)   |                  0.00363875  |                   0.00756303 |
| (128, 4096)   |                  0.200574    |                   0.75569    |
| (128, 8192)   |                nan           |                 nan          |
| (128, 16384)  |                nan           |                 nan          |

TODO: run this on A100


## Problem (torch_compile):  Torch Compile (2 points)
**(a) Extend your attention benchmarking script to include a compiled version of your PyTorch implementation of attention, and compare its performance to the uncompiled version with the same configuration as the pytorch_attention problem above.**

Deliverable: A table comparing your forward and backward pass timings for your compiled attention module with the uncompiled version from the pytorch_attention problem above.

On Laptop

| d_k,seq_len   |   fw_nocompile |   bw_nocompile |   fw_docompile |   bw_docompile |
|:--------------|---------------:|---------------:|---------------:|---------------:|
| (16, 256)     |           0.94 |           1.74 |           0.68 |           1.1  |
| (16, 1024)    |           9.65 |          17.55 |           1.59 |           2.29 |
| (16, 4096)    |          67.88 |         147.96 |          16.57 |          27.45 |
| (32, 256)     |           1.17 |           1.47 |           0.4  |           1.17 |
| (32, 1024)    |           3.66 |           7.2  |           1.13 |           2.39 |
| (32, 4096)    |         198.74 |         755.47 |          17.71 |          30.77 |
| (64, 256)     |           0.78 |           1.25 |           0.75 |           0.85 |
| (64, 1024)    |           3.5  |           7.13 |           1.39 |           2.37 |
| (64, 4096)    |         198.17 |         744.46 |          15.55 |          30.79 |
| (128, 256)    |           0.59 |           1.23 |           0.91 |           1.22 |
| (128, 1024)   |           3.64 |           7.56 |           1.48 |           2.88 |
| (128, 4096)   |         200.57 |         755.69 |          21.17 |          38.83 |

TODO: on A100

**(b) Now, compile your entire Transformer model in your end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps?**

Deliverable: A table comparing your vanilla and compiled Transformer model.

TODO


## Problem (flash_forward):  FlashAttention-2 Forward Pass (15 points)
**(a) Write a pure PyTorch (no Triton) autograd.Function that implements the FlashAttention-2 forward pass. This will be a lot slower than the regular PyTorch implementation, but will help you debug your Triton kernel. Your implementation should take input 𝑸, 𝑲, and 𝑽 as well as a flag is_causal and produce the output 𝑶 and the logsumexp value 𝐿. You can ignore the is_causal flag for this task. The autograd.Function forward should then save 𝐿, 𝑄, 𝐾, 𝑉 , 𝑂 for the backward pass and return 𝑂. Remember that the implementation of the forward method of autograd.Function always takes the context as its first parameter. Any autograd.Function class needs to implement a backward method, but for now you can make it just raise NotImplementedError. If you need something to compare against, you can implement Equation 4 to Equation 6 and Equation 12 in PyTorch and compare your outputs. The interface is then `def forward(ctx, Q, K, V, is_causal=False)`. Determine your own tile sizes, but make sure they are at least of size 16 × 16. We will always test your code with dimensions that are powers of 2 and at least 16, so you don’t need to worry about out-ofbounds accesses.**

Deliverable: A torch.autograd.Function subclass that implements FlashAttention-2 in the forward pass. To test your code, implement [adapters.get_flashattention_autograd_function_pytorch] . Then, run the test with uv run pytest -k test_flash_forward_pass_pytorch and make sure your implementation passes it.

(b) Write a Triton kernel for the forward pass of FlashAttention-2 following Algorithm 1. Then, write another subclass of torch.autograd.Function that calls this (fused) kernel in the forward pass, instead of computing the result in PyTorch. A few problem-specific tips:

• To debug, we suggest comparing the results of each Triton operation you perform with the tiled PyTorch implementation you wrote in part (a).
• Your launch grid should be set as (𝑇𝑞 ,batch_size), meaning each Triton program instance will load only elements from a single batch index, and only read/write to a single query tile of 𝑸, 𝑶, and 𝐿.
• The kernel should only have a single loop, which will iterate key tiles 1 ≤ 𝑗 ≤ 𝑇𝑘.
• Advance block pointers at the end of the loop.
• Use the function declaration below (using the block pointer we give you, you should be able to infer the setup of the rest of the pointers)

```python
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
 
```