# Design Documentation

[中文](<../zh/README.md>) | [Français](<../fr/README.md>)

This directory explains the design of the Timestral C++ inference library in the order "what the model computes, how inference saves work, how backends compute." Usage and build instructions are in the [project README](<../../README.md>), example parameters and run instructions in the [example guide](<../../example/README.md>), and numeric evidence in local validation records (not committed).

## Changelog

[Version history](<changelog.md>) lists concise changes by version. Append new versions at the end.

## Suggested Reading Order

1. [Model Architecture and Repository Modules](<architecture.md>): introduces patch embedding, causal normalization, time and variate attention axes, masks, partial RoPE, gated attention, SwiGLU, and the quantile head using Timestral-5M's real dimensions, then explains repository module boundaries and link-time backend selection.
2. [Inference-Side Design](<inference.md>): how a request goes through scheduling, block autoregression, and post-processing; why paged KV caching works on the standard path but not on the conditional path; and the frozen-prefix sliding-window experiment at the example level.
3. [Compute Backends](<backends.md>): implementation approach of reference, Eigen, and ytensor, tolerance criteria for accuracy acceptance, and measured performance trade-offs.
4. [BLAS Backend](<blas.md>): the optimization tree from request level to microkernel, deriving pre-packing, cache blocking, online softmax, block masks, and 2^x vector exp, with performance criteria and failed experiments.

Concepts in earlier documents are used directly in later ones; for example, the inference document's cache conditions depend on the architecture document's mask rules. If you only care about one backend, read the modules section of the architecture document first, then jump to the corresponding document.

For specific interface parameters, return values, exceptions, and scheduling policies, see the [API Reference](<api.md>) directly.

## Cross-Platform Builds

[Linux Compatibility Fixes and Validation Boundaries](<linuxCompatibility.md>) records item-by-item fixes for public headers, thread dependencies, ytensor local C++20/ISA guards, UTF-8 paths, and dual Vulkan compilers, along with local results and pending Linux CI. Known numeric failures are not ignored because a build passes.

## GPU Development Milestones

[Vulkan GPU Backend](<vulkan.md>) covers the ten-operator SPIR-V library, ABI manifest and reflection tests, resident executor, model backend, and 32k measured performance. The current GPU backend is Vulkan only; see [GPU Backend Scope](<gpuScope.md>) for why the old OpenGL/OpenCL components were removed.

[Vulkan Portability and Parallel Budget](<vulkan-portability.md>) records the Intel iGPU target, local ggml partial review, GPU logic window configuration, 52-module dispatch contract, strict budget cost, and pending Intel hardware validation. Decode is accepted on latency/traffic, not the development device's 3 TFLOPS denominator.

## Performance and Quality Reproduction

[Benchmark Protocol and Raw Data](<benchmark.md>) records 16 recipes and per-variable-count prefill results. The main README keeps only the single-variable table, speed chart, and one RMSE set; multi-variable tables and AR default-step verification are concentrated here. Data was measured under conditions allowing light daily load with no external heavy load; the failing scope of extended cases is listed separately.

## Figures

Figures are in the repository root [assets](<../../assets>) directory. SVGs with the arch-, infer-, and backend- prefixes correspond to the first three documents; blas- prefixed SVGs correspond to the BLAS tutorial; SWA scan charts and raw CSVs come from the sliding-window experiment.
