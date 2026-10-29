# Timestral Model Architecture and Repository Modules

This document introduces the Timestral network structure in the order a single prediction passes through it, and then explains how the C++ repository is split into modules. Dimensions are taken from Timestral-5M, from the config.json of the locally released model (model files are not under version control); the total number of weights computed from these dimensions is 5208960, which matches the F32 value count of the 5M weights in the local weight records. The network semantics are ported from the upstream fixed baseline a534bdbd; the correspondence is documented in the file headers of each source file.

## The Problem to Solve

The input to time series forecasting is the historical readings of one or several channels, and the output is the values for a number of future steps. A foundation model additionally has to face three things: the units of different series can differ by several orders of magnitude, the history contains missing measurements, and what the caller needs is often not just a point forecast but quantile intervals (calibration on arbitrary data is not guaranteed).

The most naive approach is to treat each sample point as a token and autoregress them one point at a time in the style of a language model. This road quickly hits a bottleneck on time series: the context of the 5M configuration is 32768 sample points, and per-point tokens would push the attention matrix up to a 32768×32768 scale; forecasting 512 steps would require 512 forward passes; and point forecasts cannot express uncertainty either.

Timestral's answer can be summarized in four points, each unfolded below:

- Patching: every 32 sample points and their missing flags are compressed into one token, turning a 32768-point context into 1024 patch tokens.
- Causal normalization: each patch is scaled by the cumulative mean and standard deviation up to the end of the patch, then asinh is applied, eliminating unit differences without peeking into the future.
- Independent forecast queries: each patch to be forecast is represented by a learned query vector, and a single forward pass outputs all 21 quantiles of the patch's 32 steps.
- Two kinds of attention axes: the time axis attends among patches of the same channel, the variate axis attends among different channels at the same moment, and the two alternate according to the configuration.

## The 5M Configuration

| Field | Value | Meaning |
| --- | --- | --- |
| `patch_size` | 32 | number of sample points P per patch |
| `d_model` | 192 | hidden dimension |
| `num_heads` | 3 | number of attention heads, per-head dimension 64 |
| `mlp_hidden_dim` | 512 | SwiGLU intermediate dimension |
| `block_schedule` | 8 entries | time×3, time_variate, time×3, time_variate |
| `quantile_knots` | 21 | 0.01, 0.05 to 0.95 in steps of 0.05, 0.99 |
| `block_size` | 32 | maximum number of patches decoded per forward pass |
| `context_length` | 32768 | number of context sample points, i.e. 1024 patches |
| `rope_partial_factor` | 0.5 | of the 64 per-head dimensions, 32 participate in the rotation |
| `time_attn_gate`, `variate_attn_gate` | `sdpa_sigmoid` | attention output multiplied by a sigmoid gate |
| `quantile_width_factor` | 3.0 | spacing factor of the quantile head |
| `use_asinh` | true | asinh applied after normalization |
| `attend_missing_patches` | true | missing patches remain attendable once the row has produced its first observation |
| `attn_bias`, `mlp_bias` | false | attention and MLP projections carry no bias |

Field reading and validation are in [config.hpp](<../../timestral/utils/config.hpp>); unknown fields and missing required fields are both rejected.

![Structure of one Timestral-5M forward pass](<../../assets/arch-overview.svg>)

## Causal Patch Normalization

Each row (one channel) performs a Welford accumulation along the time dimension, accumulating only observed points. The state at the end of patch p gives the loc and scale shared by all points of that patch:

$$
\mu_p=\frac{\text{sum}}{\text{count}},\qquad
\sigma_p=\max\left(\sqrt{\frac{M_2}{\max(\text{count}-1,1)}},\ \text{minimum\_scale}\right)
$$

$$
\tilde{x}_t=\operatorname{asinh}\left(\frac{x_t-\mu_p}{\max(\sigma_p,\varepsilon_{\text{FP32}})}\right)
$$

The normalized value of a missing point is set to 0; the missingness information is carried separately by an indicator bit. The accumulators are stored in double in [patch.cpp](<../../timestral/utils/patch.cpp>); source comments explain that the accumulation order, asinh, and minimum_scale are consistent with the upstream causal_patch_loc_scale.

The key here is "causal": the scaling of patch p depends only on the observations of that patch and those before it. Appending data later does not change the input of earlier patches; this is one of the prerequisites for the paged KV cache discussed later. Each page therefore stores the count, sum, and M2 at the end of the patch, and incremental decoding continues accumulating from there without rescanning the entire prefix; see [inference design](<inference.md>).

The known-future covariate rows are the exception: they are normalized by the variance over the whole window (context plus horizon); binary rows whose values are only 0 and 1 have loc=0 and scale=1 fixed, and asinh is not applied; the implementation is in [transformer.cpp](<../../timestral/utils/transformer.cpp#L502-L544>).

## Patch Embedding and Forecast Queries

The input of each patch is 2P=64 dimensions: the first 32 are normalized values, the last 32 are missing indicators (1 for missing). The embedding layer `representation.patch_proj` is a two-layer MLP with a residual branch; all three projections carry bias:

$$
h=W_o\,\operatorname{SiLU}(W_h u+b_h)+b_o+W_r u+b_r,\qquad u\in\mathbb{R}^{64},\ W_h\in\mathbb{R}^{768\times 64},\ W_o\in\mathbb{R}^{192\times 768}
$$

Patches to be forecast do not pass through the embedding layer; instead a single learned 192-dimensional vector `representation.forecast_query` is placed uniformly. Without covariates, a row's token sequence is N context patches followed by H query slots; the query slots are distinguished from each other by positional encoding and attention. This is consistent with the description in the official repository from the records: the forecast comes from independent prediction tokens that read the causally encoded history (that description comes from a search excerpt of the official repository; the full online article was not fetched).

When covariates are present, the sequence becomes dual-stream: the encoding stream contains all patches of context plus horizon (known-future values also enter the encoding), and the decoding stream places one query for each encoding slot; only the last H decoding outputs are taken into the quantile head, corresponding to `encodedCount` and `decodedCount` in the code.

## Block Schedule and the Two Attention Axes

Each entry of `block_schedule` expands into one or two attention sublayers, see [patch.cpp](<../../timestral/utils/patch.cpp#L20-L34>):

| Value | Sublayer order |
| --- | --- |
| `time` | time |
| `variate` | variate |
| `time_variate` | time, then variate |
| `variate_time` | variate, then time |

5M has 8 layers in total; after expansion there are 8 time sublayers and 2 variate sublayers. Each sublayer has its own `attn_norms`, `wQKV`, `wO`, and `_gate`; each layer has a single MLP.

Time-axis attention runs across slots within the same row, carrying the temporal dependency. Variate-axis attention runs across rows within the same slot, only allowing rows of the same `groupIds` to see each other, carrying the information exchange between channels; it does not need K/V across patches. The paged cache therefore only stores the K/V of time sublayers, 8 per page in 5M.

## Causal Mask and Missing Patches

The time-axis mask is given by `timeAllowed`, see [transformer.cpp](<../../timestral/utils/transformer.cpp#L706-L739>). First, a valid key is defined: a patch is valid if it has an observation, or if attending to missing is allowed and the row has already produced an observation earlier; valid keys can be read by later slots ([buildKeyValid](<../../timestral/utils/patch.cpp#L167-L188>)). Without covariates the rules are:

- An encoding query reads only valid encoding keys no later than itself; itself is always visible.
- A decoding query reads all valid context encoding keys, and decoding slots no later than itself.
- An encoding query does not read the decoding stream.

![Example of the time-axis mask](<../../assets/arch-mask.svg>)

With covariates, three additional rules apply: the encoding queries of known rows are bidirectionally visible in time; queries in the decoding stream that sit in the context segment do not read the encoding stream; decoding queries of the future segment read only the encoding keys of the context segment. The variate axis has another "firewall": covariate rows cannot read target rows, known rows cannot read non-known rows, and information flows only from covariates to the target.

The bidirectional visibility of known rows directly affects caching: known future values can change the hidden states of historical patches. The fixed upstream probe in the records shows that merely adding 3 to one known future value changes the target history hidden state after layer 4; see the known-future verification in the local acceptance records. So the conditioned path does not reuse historical KV across requests.

## Partial RoPE

Only time sublayers apply rotary positional encoding to Q and K; V and variate sublayers are not rotated. The dimensions participating in the rotation are the per-head dimension times `rope_partial_factor`; in 5M that is 64×0.5=32. The pairing is not between adjacent dimensions, but dimension j with dimension j+32, j ranging from 0 to 15:

$$
\theta_j=10000^{-2j/32},\qquad
\begin{bmatrix}x'_j\\ x'_{j+32}\end{bmatrix}=
\begin{bmatrix}\cos p\theta_j & -\sin p\theta_j\\ \sin p\theta_j & \cos p\theta_j\end{bmatrix}
\begin{bmatrix}x_j\\ x_{j+32}\end{bmatrix}
$$

![Partial RoPE layout](<../../assets/arch-rope.svg>)

The implementation is in [kernels.cpp](<../../timestral/reference/kernels.cpp#L128-L162>). The fused QKV of the released weights is laid out as (qkv, headDim, head), while attention needs (head, headDim); the reshuffle is done in `prepareQkv`, see [reference/model.cpp](<../../timestral/reference/model.cpp#L35-L60>).

Position p is the slot index minus the index of the first patch with an observation in the same group. Without covariates, query slots are numbered continuing after the context; with covariates, the positions of the decoding stream are aligned with the encoding stream, i.e. slot minus the number of encoding slots. `Settings::continuousRope` can replace integer slots with real-valued coordinates derived from real time; this is an experimental option off by default, and the samples in the records showed no accuracy improvement.

## Gated Attention

Each attention sublayer uses a pre-norm structure:

$$
\hat{x}=\operatorname{RMSNorm}(x),\quad
o=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{64}}+M\right)V,\quad
x\leftarrow x+W_O\left(o\odot\sigma(W_g\hat{x})\right)
$$

The gate `_gate` reads the same normalized input as QKV and acts elementwise on the concatenated 192-dimensional attention output. When the configuration is `none` there is no gate, and the weight contract does not require `_gate` either. The epsilon of RMSNorm is 1e-6.

## SwiGLU MLP

The MLP is also pre-norm: `w12` projects 192 dimensions to 1024, multiplies the first half elementwise by SiLU(second half) to get 512 dimensions, then `w3` projects back to 192 dimensions and adds to the residual. This order, "of the two halves, only the second half goes through SiLU", matches upstream; swapping them would give entirely different results, see [transformer.cpp](<../../timestral/utils/transformer.cpp#L35-L51>).

## Quantile Head

After the last layer comes `out_norm`; the hidden states of the decoding slots pass through `head.projection` (a residual MLP isomorphic to the embedding layer, 192→768→672, without bias), and are reshuffled into raw values r of 32 steps × 21 quantiles. To guarantee quantile monotonicity, the median channel r_m is output directly, and the other channels accumulate spacings via softplus:

$$
q_j=r_m-\sum_{i=j}^{m-1}w\,(\tau_{i+1}-\tau_i)\,\operatorname{softplus}(r_i),\quad j<m
$$

$$
q_j=r_m+\sum_{i=m+1}^{j}w\,(\tau_i-\tau_{i-1})\,\operatorname{softplus}(r_i),\quad j>m
$$

where w is `quantile_width_factor` and τ are the quantile points. The results are clamped to the finite range of FP32. `predictBlock` then does the inverse normalization: first sinh, then multiply by the scale of the last context point and add loc; infinite values from sinh overflow are replaced by the maximum or minimum of the finite quantiles at that position, and 0 if neither is finite.

## Parameter Count Audit

Counting item by item according to the structure above verifies that the understanding of the structure has no missing items:

| Component | Weights | Parameter count |
| --- | --- | ---: |
| patch embedding | three projections of patch_proj, with bias | 210048 |
| forecast query | forecast_query | 192 |
| attention sublayers ×10 | attn_norm, wQKV, wO, _gate | 1845120 |
| MLP ×8 | mlp_norm, w12, w3 | 2360832 |
| output normalization | out_norm | 192 |
| quantile head | three projections of head.projection | 792576 |
| total | | 5208960 |

The [Transformer constructor](<../../timestral/utils/transformer.cpp#L146-L214>) validates the name, shape, and finiteness of every weight according to the same checklist; loading is rejected if any weight is extra or missing.

## C++ Repository Modules

~~~text
timestral/
├── timestral.hpp/.cpp          # public entry point, forwards to the internal scheduling service
├── utils/                      # backend-agnostic infrastructure and shared model semantics
│   ├── service.hpp/.cpp        # request queue, three scheduling strategies, shared prefix tree
│   ├── engine.hpp/.cpp         # tensor-level engine: chunked autoregression, refine, post-processing orchestration
│   ├── condition.hpp/.cpp      # grid alignment and segmented conditional completion of interleaved observations
│   ├── forecast.hpp/.cpp       # request validation and result extraction
│   ├── time.hpp/.cpp           # from real time to real-valued patch coordinates
│   ├── transformer.hpp/.cpp    # shared model orchestration, masks, paged execution
│   ├── patch.hpp/.cpp          # causal normalization, patch assembly, positions, page keys
│   ├── cache.hpp/.cpp          # page pool, FNV page keys, references and LRU candidates
│   ├── model.hpp               # backend interface timestral::Model
│   ├── backend.hpp             # declaration of loadBackendModel
│   ├── attention.hpp           # attention views and mask descriptions
│   └── other base files        # tensor, config, json, safetensors, postprocess, stats
├── reference/                  # scalar kernels, the default backend for non-x86 targets
├── eigen/                      # Eigen optional backend
├── ytensor/                    # ytensor optional backend
└── blas/                       # home-grown BLAS optional backend
~~~

The responsibility boundaries follow the repository's internal code conventions: the root of `timestral/` contains only the two entry-point files; `utils/` carries everything backend-agnostic, including scheduling, caching, normalization, post-processing, and statistics; a backend directory contains only that backend's own linear algebra and kernels, not cache management or post-processing. There is no separate ops layer; the "operators" shared across backends are actually model semantics, which belong to `utils`.

![Module split and dependency direction](<../../assets/arch-modules.svg>)

The public header [timestral.hpp](<../../timestral/timestral.hpp>) exposes only `Sample` and `Settings`, `Request`, `Result`, and `Policy` nested inside `Timestral`; internal types appear only as the forward declaration `Service`. Tensors, the page pool, and the role table all stay internal; callers deal only with sample points.

## Backends Are Chosen at Link Time

Backend selection does not go through macros. [backend.hpp](<../../timestral/utils/backend.hpp>) declares only one factory function `loadBackendModel`; each of the four backend directories has a `backend.cpp` giving its definition, and CMake adds only one of them to the library target at configure time:

~~~cmake
set(TIMESTRAL_SELECTED_BACKEND timestral/reference/backend.cpp)

if(TIMESTRAL_USE_EIGEN)
    set(TIMESTRAL_SELECTED_BACKEND timestral/eigen/backend.cpp)
endif()
~~~

The above is an excerpt of [CMakeLists.txt](<../../CMakeLists.txt#L19-L35>); the branches for ytensor and BLAS work the same way; enabling any two optional backends at the same time exits with FATAL_ERROR during the configure stage. The result: the whole build has a single definition of `loadBackendModel`, no backend conditional compilation is visible in the public header or scheduling code, and the default build does not include the Eigen or ytensor headers. The price is that only one backend can be built at a time; comparing backends requires configuring separate build directories.

## Data Flow of a Single Prediction

1. When `Timestral` is constructed, `loadEngine` reads config.json and model.safetensors, calls the linked `loadBackendModel` to get a read-only model, and then starts the working threads of `Service`.
2. `submit` completes all validation on the calling thread: history no shorter than 2 patches, strictly increasing timestamps, inferred sampling interval and forecast steps, trimming the history to `maxContext`, and deciding whether the paged grid can be used.
3. The worker thread takes one round of requests according to the scheduling strategy; those hitting prefix pages go through the paged path, the rest are fully recomputed or cold-batched.
4. [Engine](<../../timestral/utils/engine.cpp>) left-pads the context to a multiple of the patch size and loops over `patchBatch` patches per block: a forward pass produces quantiles, an advancing refine is done, and the target median is fed back as the context of the next block.
5. After all blocks finish, range clipping, symmetric averaging, and non-negativity clipping are applied in turn, then the result is assembled into a `Result`, with timing and page-hit statistics attached.

For the scheduling, caching, and post-processing details of steps 2 through 5, see [inference design](<inference.md>).

## Trade-offs

The model orchestration is written only once. [Transformer](<../../timestral/utils/transformer.hpp>) holds all the logic of the embedding, masks, paging, and the quantile head; reference and BLAS only override the narrow hooks of projections, RMSNorm, QKV layout, and attention, so a change to masks or paging does not have to be synchronized across multiple backends. Eigen and ytensor are backends integrated earlier; they still each keep a complete orchestration of their own, replacing only the matrix multiplication with their own library and calling the reference kernels for elementwise operators; this duplication is a known historical cost, and when changing shared semantics all three places must be checked together.

The `split_context_horizon_attn` field is parsed and stored, but no computation branch reading it was found in the current C++ source; the miniature-model comparison with "split off" in the records passed, but the exact semantics of this switch upstream have not yet been checked line by line, so this is marked as unverified.

The lesson to be drawn from this structure: whether a model can do incremental inference depends mainly on its dependency structure, not on the engineering implementation. Causal normalization, the causal time mask, and independent forecast queries make the K/V of historical patches depend only on the history itself, which is what makes the paged cache meaningful; the bidirectional attention of known rows and the whole-window statistics break this property, and the corresponding path can only be fully recomputed. Understanding the mask determines, before optimizing any single GEMM, what the inference side can do at all.
