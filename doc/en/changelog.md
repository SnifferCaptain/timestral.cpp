# Changelog

[中文](<../zh/changelog.md>) | [Français](<../fr/changelog.md>) | [Documentation index](<README.md>)

Use a second-level heading for each version and a table of concise changes; notes are optional. Append new versions at the end in release order. Keep versions and row order aligned across languages.

## 1.1.0

Current working-tree changes; not yet released.

| Type | Change | Notes |
| --- | --- | --- |
| Changed | Default to single-threaded BLAS on x86 | TIMESTRAL_USE_REFERENCE forces reference. |
| Optimized | Default to 2048 KV pages, allocated on demand | cachePages=0 disables caching. |
| Added | Default an omitted horizon to one block AR round | 512 samples by default for 5M. |
| Added | Add session IDs, append requests and close | maxSessions=16; 0 disables sessions. |
| Added | Add predicates on top of native causal masks | Eigen/ytensor/Vulkan use slower reference. |
| Fixed | Left-pad partial history to match full forecasts |  |
| Fixed | Keep long-session patch boundaries; buffer tails | bufferedSamples reports pending samples. |
| Added | Set the short-history prefill limit to 512 | fullPrefillThreshold=0 fixes boundaries. |
| Fixed | Keep forecasts just after the latest sample | Buffered values are not yet used. |
| Optimized | Cache and reuse negated history for averaging | Page counts cover the positive prefix only. |
| Fixed | Reuse mirrored KV in shared batch decoding | Divergent tails are still computed separately. |
| Fixed | Honor the default horizon in conditional forecasts | horizonSeconds=0. |
| Fixed | Retain needed conditional past and known future | Includes large absolute timestamp boundaries. |
| Fixed | Release old named views to prevent a full pool | Other sources keep their references. |
| Fixed | Roll back failed prefill and discard partial pages | Existing in-use prefixes are preserved. |
| Fixed | Close queued sessions with an explicit streamId | Running computations are not preempted. |
| Fixed | Preserve history-update and request-queue order | Prevent concurrent snapshots from reordering. |
| Fixed | Build session history fully before replacing it | Allocation failure leaves no partial history. |
| Fixed | Guard context trimming and session ID overflow |  |
| Fixed | Check KV page geometry and page-index bounds | Invalid page indices throw an exception. |
| Fixed | Mark pages occupied only after KV allocation |  |
| Fixed | Validate intervals and step-count conversions | Also avoid overflow in the median average. |
| Fixed | Align ytensor projection rows for cached accuracy | Pad at most 5 work rows, not time samples. |
| Experimental | Add frozen KV with an attention sink page | reference/BLAS only; not a default path. |
| Documentation | Move design docs into language directories | Chinese design documents are in doc/zh. |
| Documentation | Add a [French README](<../../README_fr.md>) and language switches |  |
| Documentation | Add changelogs and language-specific doc indexes | Append later versions at the end. |
| Documentation | Correct the outdated README CI status | CI verifies builds and links only. |
| Tests | Cover sessions, conditions, errors and mirrored KV | Affected paths checked on five backends. |
| Known limitation | ytensor 8-channel RoPE still exceeds tolerance | Tolerance unchanged; session fix is separate. |
