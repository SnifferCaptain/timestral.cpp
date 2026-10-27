# Third-Party Sources

## Timestral

Model semantics and the ported implementation derive from [Mistral AI Timestral](<https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff>), pinned at commit a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff. The upstream license is Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0); the repository-wide license is [LICENSE](<LICENSE>), and the upstream origin is recorded here.

The following files are ported or rewritten along the upstream model structure and computation flow, with the source marked in each file:

- [Config implementation](<timestral/utils/config.cpp>): upstream config fields and inference validation.
- [Reference operators](<timestral/reference/kernels.cpp>), [shared model orchestration](<timestral/utils/transformer.cpp>) and [reference numeric hooks](<timestral/reference/model.cpp>): layers, model, masking and normalization.
- [Inference flow](<timestral/timestral.cpp>): the inference entry, class encoding, causal advance and post-processing.

This is a C++ rewrite, not a packaged upstream Python runtime. The default library, examples and tests do not link PyTorch, NumPy, safetensors or huggingface_hub; the reference-data generation tool (local tests/generateReference.py, not shipped) is developer-optional and depends on those Python packages plus the pinned upstream source.

## Weights and Test Data

Released Mistral weights are under a separate [Modified MIT license](<https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE>) and are not covered by the code licenses above. This repository does not distribute released weights and does not download models automatically.

The tiny model in the [test data notes](<tests/README.md>) is built from a fixed random seed; its parameters are random test inputs, not slices, quantizations or conversions of released Mistral weights.

## Optional-Backend Third-Party Headers

The [3rd directory](<3rd/README.md>) holds the pinned third-party headers of the two optional backends, included only when the matching CMake option (TIMESTRAL_USE_EIGEN, TIMESTRAL_USE_YTENSOR) is explicitly enabled; the default build does not reference them.

- [Eigen 3.4.0](<https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz>): header tree at [3rd/eigen](<3rd/eigen>), MPL-2.0, full license texts kept with the tree.
- [ytensor single header](<https://raw.githubusercontent.com/SnifferCaptain/ytensor/main/single-header/ytensor_single.hpp>): renamed to [3rd/ytensor.hpp](<3rd/ytensor.hpp>), generated 2026-07-31, version 0.14; that repository's main branch has no license file, so it was vendored on instruction with no license declared; its bfloat16 subcomponent is Apache-2.0 (from openvino). Enabling this backend requires C++20.

## BLAS Design References

The pack/tile/microkernel responsibility split of the hand-written BLAS follows the [BLIS 2.0](<https://github.com/flame/blis/tree/e8566eb3e773fb54d11b33e371d13f22d2941e50>) [design docs](<https://github.com/flame/blis/blob/e8566eb3e773fb54d11b33e371d13f22d2941e50/docs/KernelsHowTo.md>) and [Haswell SGEMM](<https://github.com/flame/blis/blob/e8566eb3e773fb54d11b33e371d13f22d2941e50/kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c#L80>); the current code uses independently written C++ intrinsics and does not copy or link BLIS source. BLIS is BSD-3-Clause, and its per-file copyright notices and [full license](<https://github.com/flame/blis/blob/e8566eb3e773fb54d11b33e371d13f22d2941e50/LICENSE>) apply to any future real port; the absence of a linked dependency does not waive source or binary redistribution obligations.

The online softmax, block mask and shared views also studied the [ytensor Qwen3 example](<https://github.com/SnifferCaptain/ytensor/tree/6f1a4d25f97d753f70d9ea61a9ca181fc835ebd9/example/qwen3>); only the algorithm and dependency analysis were borrowed, and its license-unclear code was not copied into the hand-written BLAS. Implementation and quantization-path boundaries are in the local phase plan.

## Cephes Exponential Polynomials

The six exp2f polynomial coefficients in the [vector exp](<timestral/blas/elementwiseAvx2.cpp>) come from Cephes Math Library Release 2.2 (June 1992) by Stephen L. Moshier; exp2f.c from the [original archive](<https://netlib.org/cephes/single.tgz>) was read and the coefficients verified. Base change, FMA reduction, exponent bit splicing and the AVX2 loops are this repository's own implementation; no Cephes library is linked. The original file copyright notice and the [Netlib distribution notice](<https://netlib.org/cephes/readme>) are kept in the [file header](<timestral/blas/elementwiseAvx2.cpp>) and in [LICENSES/LICENSE.cephes](<LICENSES/LICENSE.cephes>). The notice permits free use with no support or guarantee; it is not relabeled BSD or MIT.

## Style Reference

Directory layout and prose style follow [tdt-nav-kit](<https://github.com/T-DT-Algorithm-2026/tdt-nav-kit/tree/814d5ce0237acc7faa8c070e5c5e61e2ddc81996>); none of its navigation algorithms, assets or external dependencies were copied.

The repository-wide license is [LICENSE](<LICENSE>). Each third-party component keeps its own license with its files: Eigen full texts in 3rd/eigen/, the upstream Apache-2.0 text in [LICENSES/LICENSE.timestral](<LICENSES/LICENSE.timestral>), and the Cephes notice in the source file header and [LICENSES/LICENSE.cephes](<LICENSES/LICENSE.cephes>).
