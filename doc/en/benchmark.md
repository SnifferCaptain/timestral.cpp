# Performance and Quality Benchmarks

## Data and Measurement Conditions

The following raw data and their source records are kept in the maintainer's local workspace and are not distributed with the repository; the numbers on this page are the recorded results of those files.

- Prefill raw CSV `backend-prefill-20261025-v3.csv`: the original measurements are carried over; no numbers were refilled
- Raw CSV of the 16k-context to 1k decode `backend-decode-1k-20261026.csv`
- 1k recipe, binary hashes, and integrity record `backend-short-20261026.json`
- 0-to-4k chunk-AR per-length CSV `block-ar-4k-20261026.csv`
- 4k official comparison, model/program/data hashes `block-ar-4k-20261026.json`
- Aggregate RMSE and 4k average time `block-ar-4k-summary-20261026.csv`
- Plot sources and numerical failure confirmation `block-ar-4k-summary-20261026.json`

The CPU is an AMD Ryzen 7 7735H; the GPU is the AMD Radeon(TM) Graphics of enumerated device 0. Collection allows everyday light load and does not guarantee complete idleness. Recipes run serially; no building or running of other benchmarks during the timed runs; the official Python verification runs after the timing completes.

The 1k backend collection contains 16 recipes. BLAS AVX2 includes 1, 2, and 4 threads; the other BLAS recipes without a stated thread count are 1 thread. Both the regular and the accelerated Vulkan are kept, and the 512/4096 logical dispatch budgets are also in the raw data, not corresponding to physical EU counts. The main figure shows 12 representative recipes among them; the sources of prefill and decode are recorded separately, and the new decode is not passed off as the old 16k data with a changed title.

## Inputs and Scoring

The backend comparison uses the first channel of an eight-channel signal configuration (local tool input, not checked in): real history indices 0 to 16383, future indices 16384 to 17407, single variable. ar decodes 1 patch per round, autoregressing for 32 rounds; par does a normal one-shot forward of 32 patches to get 1024 points, without calling the cross-block RoPE Parallel. Both the backend and the step-size comparisons disable symmetricAveraging/clipNonnegative, keeping the native refine and the quantile range handling.

The quality curves use a six-function configuration (local tool input, not checked in), each function as an independent single-variable request. This round uses 16k history and 4k future; the JSON provides the functions, noise, and sampling parameters; the horizon field of the old examples does not determine the actual forecast length of this tool; the 4096-point tensor shape, the raw data row counts, and the official request length are all verified separately. The six groups are: period-128 single sine, double harmonic, period-512 slow sine, period-128-and-176 mixture, 5-odd-harmonic square-wave approximation, and a 128-carrier/2048-envelope amplitude-modulated sine. Interval 0.01 s, noise standard deviation 0.03, no jitter or phase noise, seed 42 + function index × 10007. Start points 16384 and 16448; each run reads only the 16384 real observations before it; the noise-free future function ground truth is scored only and does not enter the context, mask, or feedback.

## The Official Decode Flow

Timestral-5M has patch_size=32, native block_size=32, and one complete block forward is 1024 points. The public Settings.patchBatch controls how many patches are forecast per round, allowing 1 to 32; exceeding the model block size is clamped, 0 is rejected; the default is 16. For the 4k task the aggregate RMSE is 0.283 at 16 patches and 0.768 at 32 patches, and the average time of a complete 4k decode is 224 and 166 ms respectively; see below and the aggregate CSV for the data. Between blocks the forecast median is fed back, and under the same input and budget a smaller block does not guarantee being more accurate or faster.

"par" on this page specifically means one normal 32-patch forward of the 1k task, still going through the ordinary chunk-AR implementation; requesting 4k with patchBatch=32 forecasts four blocks normally. The old cross-future-block RoPE Parallel, which froze the prefix and skipped the inter-block feedback, is kept as a historical experiment behind the C++ deprecated marker and does not enter the public predict or new figures.

The official forecast enables symmetric averaging and non-negativity clipping by default; this round disabled them uniformly to compare step sizes, which is not equivalent to the complete default post-processing. continuousRope only controls the time coordinates and is not the same feature as the deprecated cross-block RoPE Parallel.

## Short Forecast Accuracy and Block Size

The [main description](<../../README.md#short-forecasts-and-decode-block-size>) keeps only [one aggregate figure](<../../assets/block-ar-4k-summary-20261026.png>); the left is the multi-function aggregate RMSE, the right the average decode time of a complete 4k request. The step sizes are still 1, 2, 4, 8, 16, 32 patches, all using the normal chunk AR.

Each trajectory actually forecasts 4096 points, then scores cumulatively over 32, 64, …, 4096 points; the 0 on the horizontal axis is only the start point, with no fake 0-point error. The H-point aggregate RMSE is sqrt(Σ[all functions, all start points, first H points](forecast median − truth)² / Σ sample count). Currently there are 2 start points per function, so the denominator is 12H, and the functions are equally weighted at the mean-square-error level. **The RMSEs are not averaged directly, the high-error functions are not removed, and samples are not selected by different step sizes.** The blue dashed line merges the zero-forecast error by the same method. The aggregate score only summarizes this set of same-unit synthetic functions; it is not a general accuracy guarantee on real datasets; the per-function results remain in the raw CSV.

Each complete 4k request is warmed up once and timed 3 times, taking the median, with the six step-size orders rotated. The bar chart takes the arithmetic mean of the 12 median times of 6 functions × 2 start points, without resampling the same latency repeatedly by cumulative length. The timing includes predictStreamed, the private feedback page-pool construction/copy, the forecast feedback encoding, and post-processing; it does not include the initial observation prefill, loading, input generation, or scoring. factor = aggregate 1-patch time / current step-size aggregate time; 1 patch is not relabeled 1.0×; milliseconds are shown to the integer, with exact values kept in the CSV. decode_ms_4k always refers to the complete request, not the independent latency of each cumulative prefix.

The collection is 72 trajectories, 9216 cumulative metrics, and 294912 scored median points; the aggregate CSV is 6 step sizes × 128 lengths = 768 rows. The JSON ground truth, SSE, counts, and RMSE were independently recomputed, and then checked one by one against the official forecast of the same step size at the fixed upstream a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff. Both sides disable symmetric averaging and non-negativity clipping. The original atol1e-4/rtol1e-3 budget is unchanged. After a strict check of the complete run, 67/72 trajectories are within bounds; the other 5 trajectories have 54/294912 points in total exceeding the bounds, and the judgment is still not a full pass. The corresponding function/start/step-size and errors are fully preserved in the metadata; the C++ measured curves were not replaced with the official curves. Plotting uses an explicit failure confirmation, with a short warning kept in the figure; the same-named aggregate JSON records that confirmation and the source record hashes, without changing the strict-check record or its failure status. The quantile head and the other backends for this set of 4k requests were not fully verified.

The old 2k per-function data and numerical records (kept locally) are historical; they are not retitled and passed off as 4k.

## Prefill Accounting

Lengths 512, 1024, 2048, 4096, 8192, 16384, 32768; at each length 1, 2, 4, and 8 variables are measured separately. Each group uses 3 fresh empty page pools; the timing runs from appendPages until all prefix pages are ready, taking the median. Weight loading, pipeline construction, and page-pool pre-allocation are not counted.

The [single-variable results](<../../README.md#prefill-time>) are in the main description; each cell below has only one millisecond value.

## Multivariate Prefill

### 2 variables

| Sequence length | reference | Eigen 1 thread | Eigen 2 threads | ytensor 1 thread | ytensor 2 threads | BLAS AVX2 1 thread | BLAS AVX2 2 threads | BLAS AVX2 4 threads | BLAS SSE4.2 | BLAS FP16 | BLAS BF16 | BLAS INT8 | Vulkan regular | Vulkan accelerated | Vulkan accelerated 512 | Vulkan accelerated 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 99.27 | 28.96 | 19.46 | 9.38 | 9.67 | 4.23 | 3.00 | 2.45 | 8.21 | 4.12 | 4.47 | 4.45 | 4.47 | 4.22 | 26.47 | 6.34 |
| 1k | 179.93 | 61.11 | 38.69 | 18.75 | 19.03 | 8.08 | 5.38 | 4.31 | 15.51 | 8.15 | 9.35 | 8.51 | 6.14 | 5.79 | 49.41 | 10.21 |
| 2k | 358.67 | 126.98 | 90.76 | 53.94 | 42.71 | 16.97 | 13.56 | 10.60 | 32.35 | 17.65 | 18.04 | 17.46 | 10.36 | 7.84 | 47.28 | 12.02 |
| 4k | 782.63 | 313.91 | 223.25 | 130.25 | 135.44 | 39.97 | 25.58 | 20.22 | 77.62 | 40.74 | 42.58 | 42.03 | 21.14 | 14.07 | 112.73 | 24.07 |
| 8k | 1655.37 | 759.91 | 586.45 | 388.80 | 385.30 | 106.83 | 65.76 | 56.61 | 181.20 | 103.62 | 107.55 | 108.59 | 46.00 | 30.00 | 295.15 | 57.96 |
| 16k | 3889.62 | 2136.96 | 1792.12 | 1660.37 | 1555.39 | 312.53 | 187.85 | 139.89 | 470.71 | 308.60 | 320.81 | 324.57 | 112.40 | 83.38 | 860.09 | 149.63 |
| 32k | 10041.90 | 6564.70 | 5971.97 | 5756.42 | 5217.45 | 980.05 | 796.81 | 407.36 | 1386.27 | 1003.04 | 1004.14 | 999.28 | 336.24 | 269.54 | 2912.99 | 455.25 |

### 4 variables

| Sequence length | reference | Eigen 1 thread | Eigen 2 threads | ytensor 1 thread | ytensor 2 threads | BLAS AVX2 1 thread | BLAS AVX2 2 threads | BLAS AVX2 4 threads | BLAS SSE4.2 | BLAS FP16 | BLAS BF16 | BLAS INT8 | Vulkan regular | Vulkan accelerated | Vulkan accelerated 512 | Vulkan accelerated 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 187.17 | 58.52 | 37.18 | 16.20 | 16.52 | 7.63 | 4.80 | 3.88 | 15.92 | 7.68 | 8.10 | 8.03 | 6.13 | 6.00 | 48.01 | 10.32 |
| 1k | 357.64 | 119.75 | 82.08 | 37.18 | 36.73 | 15.45 | 12.90 | 8.57 | 30.52 | 15.45 | 17.92 | 16.22 | 9.68 | 6.42 | 43.53 | 10.90 |
| 2k | 737.83 | 277.12 | 202.85 | 109.67 | 90.79 | 33.90 | 23.36 | 16.92 | 65.66 | 35.73 | 35.82 | 35.75 | 19.25 | 11.92 | 94.90 | 22.46 |
| 4k | 1554.77 | 605.84 | 452.34 | 257.57 | 261.02 | 84.14 | 54.74 | 41.65 | 157.25 | 84.13 | 85.95 | 86.00 | 39.17 | 23.18 | 220.13 | 46.51 |
| 8k | 3509.92 | 1559.81 | 1180.40 | 826.31 | 769.28 | 223.00 | 139.43 | 104.73 | 365.61 | 214.35 | 223.38 | 224.15 | 85.57 | 56.13 | 573.43 | 109.56 |
| 16k | 7833.76 | 4219.80 | 3611.78 | 3189.16 | 2800.60 | 609.46 | 359.34 | 236.45 | 925.28 | 605.52 | 627.19 | 631.80 | 218.07 | 165.42 | 1720.01 | 291.73 |
| 32k | 19993.70 | 13212.52 | 12102.66 | 11393.67 | 10490.66 | 2002.03 | 1537.82 | 900.05 | 2792.45 | 1963.34 | 2076.01 | 2046.38 | 646.14 | 526.88 | 5741.83 | 915.63 |

### 8 variables

| Sequence length | reference | Eigen 1 thread | Eigen 2 threads | ytensor 1 thread | ytensor 2 threads | BLAS AVX2 1 thread | BLAS AVX2 2 threads | BLAS AVX2 4 threads | BLAS SSE4.2 | BLAS FP16 | BLAS BF16 | BLAS INT8 | Vulkan regular | Vulkan accelerated | Vulkan accelerated 512 | Vulkan accelerated 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 356.82 | 114.24 | 75.75 | 32.30 | 33.17 | 14.56 | 12.21 | 8.81 | 29.68 | 14.83 | 15.73 | 15.54 | 9.87 | 7.80 | 41.80 | 10.89 |
| 1k | 716.13 | 244.31 | 177.06 | 74.91 | 72.89 | 30.96 | 20.41 | 15.39 | 62.06 | 32.18 | 35.31 | 32.95 | 18.69 | 11.61 | 85.41 | 21.02 |
| 2k | 1461.50 | 530.54 | 362.30 | 188.60 | 178.68 | 71.78 | 46.88 | 36.77 | 138.94 | 78.51 | 74.03 | 72.89 | 36.11 | 21.65 | 183.18 | 42.40 |
| 4k | 3048.56 | 1211.35 | 895.08 | 504.51 | 504.80 | 170.25 | 115.18 | 93.95 | 309.33 | 172.51 | 177.54 | 180.27 | 71.63 | 43.02 | 437.19 | 91.02 |
| 8k | 6855.54 | 3022.08 | 2310.21 | 1804.24 | 1800.13 | 441.29 | 260.55 | 181.99 | 720.87 | 428.94 | 448.72 | 443.31 | 167.39 | 111.03 | 1183.78 | 213.05 |
| 16k | 15648.41 | 8308.61 | 7132.50 | 6194.14 | 5539.69 | 1262.54 | 732.38 | 481.52 | 1898.24 | 1265.96 | 1302.86 | 1291.84 | 433.55 | 321.66 | 3475.49 | 587.07 |
| 32k | 40023.28 | 26440.70 | 27766.18 | 22761.36 | 20549.86 | 4029.73 | 3138.08 | 1952.89 | 5701.30 | 4015.34 | 4130.65 | 4114.87 | 1610.08 | 1312.02 | 11623.59 | 1845.87 |

## Decode Accounting

The backend new figure is fixed at 16k history, 1k future, single variable, ar=1 patch and par=32 patches. Each path is warmed up once and timed 3 times, taking the median with the order alternated, excluding the initial history prefill. par calls one normal block forward; ar performs 32 decodes and 31 feedback appends; the normal chunk AR uses a private forecast trajectory and does not change the shared observation prefix.

The public API also uses Settings.patchBatch, but that value is fixed at construction, not an override per Request. A single-variable 16k history needs 512 observation pages, plus the mirror pages of symmetric averaging for a total of 1024 pages, within the default cachePages=2048. Request.horizonSeconds is in seconds; 2k at a 0.01 s interval corresponds to 20.48 s. The times in the figure should not be taken directly as the end-to-end latency of a complete public request.

## Reproduction

The matrix benchmark (local tool `example-tools/backendMatrix.cpp`) gained a short-decode mode, which by default fixes 16k/1k, ar1/par32; the old decode mode only keeps the deprecated historical RoPE experiment and no longer feeds data to new figures. Quality collection, official checking, and plotting are all local tools, not checked in.

~~~powershell
./build-blas/Release/backendMatrix.exe blas-avx2 model/timestral-5m .agents/data/example-tools/benchmark.json short-decode --threads 1
python .agents/data/example-tools/backendPlot.py .agents/data/benchmark/backend-prefill-20261025-v3.csv .agents/data/benchmark/backend-decode-1k-20261026.csv assets/backend-compare-1k-20261026.png
~~~

The figure generator rejects old 16k decode data, wrong step sizes, and missing or duplicated samples. Quality checking and plotting reject by default data that did not pass the numerical budget; only the explicit --allow-recorded-deviations allows keeping failing records and exporting a warning figure, without changing the tolerance. All the short-forecast comparisons of this round passed, so the switch was not needed. The sliding-window scan carries over the old data and was not rerun.

## Historical Diagnostics

The following keeps only the old link entries and is not the current recommended usage. The old limits of cross-platform, GPU extra 8 variables, ytensor extra 2 variables, and device-1 long queries are still in the local acceptance records; this round's short-forecast pass does not cover them.

### Long Forecast Quality Limits

See [historical long-forecast diagnostics](<forecastHistory.md#long-forecast-quality-limitations>).

### Multi-Function Long Forecast Curves

See [historical multi-function curves](<forecastHistory.md#multi-function-long-forecast-curves>).

### Single-Patch AR Supplementary Experiments

See [historical 1patch/32patch experiments](<forecastHistory.md#single-patch-ar-supplemental-experiment>).
