# API Reference

The public interface has a single entry header file [timestral/timestral.hpp](<../../timestral/timestral.hpp>); linking the CMake target timestral is enough to use it, and the default build depends only on the C++17 standard library.

~~~cpp
#include "timestral/timestral.hpp"
~~~

## timestral::Sample

A sample point; time in seconds, origin arbitrary.

**Members**
| Member | Type | Description |
| --- | --- | --- |
| time | double | timestamp, in seconds |
| value | double | reading; NaN denotes a missing measurement at that time |

## timestral::Timestral

The local Timestral inference engine: loads the model once, supports concurrent submission, the shared prefix cache is content-addressed, and zero-reference pages are evicted by LRU.

### Timestral::Timestral()

Loads the configuration and weights from a local model directory and starts the scheduling thread.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| modelDirectory | const std::filesystem::path& | - | the directory containing config.json and the weight files |
| settings | Settings | default aggregate initialization | instance configuration fixed after construction |
| policy | Policy | Policy::Fifo | the initial scheduling strategy |

**Throws**
| Exception | Condition |
| --- | --- |
| std::runtime_error | missing directory, configuration, or weights; illegal fields |

**Example**
~~~cpp
timestral::Timestral model{"model/timestral-5m"};
~~~

No copy constructor or copy assignment is provided; the destructor stops the scheduling thread, and the futures of not-yet-started requests end with std::runtime_error.

### Timestral::predict()

Submits one forecast and waits for the result, equivalent to submit immediately followed by get.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| request | Request | - | the request content, see the Request member table |

**Returns**
| Type | Description |
| --- | --- |
| Result | the forecast result |

**Throws**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | illegal request parameters |
| RequestSuperseded | invalidated by a newer same-source request or abort under the Mailbox policy |

**Example**
~~~cpp
timestral::Timestral model{"model/timestral-5m"};
std::vector<timestral::Sample> history;
for (std::size_t t = 0; t < 128; ++t)
    history.push_back({static_cast<double>(t) * 0.01, std::sin(static_cast<double>(t) * 0.05)});
timestral::Timestral::Request request;
request.variables = {history};
request.sampleInterval = 0.01;
request.horizonSeconds = 0.32;
auto result = model.predict(request);
for (std::size_t t = 0; t < result.steps(); ++t)
    std::cout << result.time(t) << ',' << result.median(t) << '\n';
~~~

### Timestral::submit()

Submits one forecast and returns immediately; parameter validation and session appending are done on the submitting thread.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| request | Request | - | an empty streamId is treated as an anonymous request, with the service assigning an identifier internally |

**Returns**
| Type | Description |
| --- | --- |
| std::future<Result> | yields the result once ready; get() throws RequestSuperseded when invalidated, and std::runtime_error on engine failure |

**Throws**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | thrown on the submitting thread when the request parameters are illegal, the session does not exist, or a new reading is not later than the session tail; the request does not enter the queue and the session is unchanged |

### Timestral::abort()

Ends one source: invalidates its not-yet-started requests and releases its held prefix-page references.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| streamId | const std::string& | - | the source identifier; the futures of the invalidated requests throw RequestSuperseded |

**Note**

Pages still referenced by other sources continue to be shared; pages with no references have their page keys detached and are released immediately.

### Timestral::close()

Deletes the history saved by one session and invalidates that session's not-yet-started requests.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| session | const std::string& | - | the session identifier, taken from Result::session; ignored when it does not exist |

**Note**

The KV pages a session has used are not released immediately; they stay in the LRU to be evicted on demand.

### Timestral::setPolicy()

Switches the scheduling strategy; only affects requests that have not been scheduled yet.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| policy | Policy | - | when switching to Mailbox, only the last not-yet-started same-source request in the queue is kept |

### Timestral::policy()

The current scheduling strategy.

**Returns**
| Type | Description |
| --- | --- |
| Policy | the current strategy |

### Timestral::pendingRequests()

The number of requests in the queue that have not started executing.

**Returns**
| Type | Description |
| --- | --- |
| std::size_t | number of requests awaiting execution |

### Timestral::patchSize()

The model's patch length; 32 for the 5M model.

**Returns**
| Type | Description |
| --- | --- |
| std::size_t | number of sample slots contained in a patch |

### Timestral::contextLength()

The longest context the model supports.

**Returns**
| Type | Description |
| --- | --- |
| std::size_t | the context upper bound, in steps |

### Timestral::Policy

The scheduling strategy enum.

| Value | Description |
| --- | --- |
| Fifo | execute in submission order |
| Mailbox | keep the latest pending request of the same source; older same-source requests are invalidated |
| Continuous | automatic batching, constrained by Settings::maxBatch and maxVariables |

### Timestral::Settings

Instance configuration, read at construction and immutable afterward. With all defaults it is the complete attention consistent with upstream, with no sliding-window approximation.

**Members**
| Member | Type | Default | Description |
| --- | --- | --- | --- |
| maxContext | std::size_t | 0 | upper bound on context samples; 0 means use the model's contextLength; the excess drops the earliest readings |
| patchBatch | std::size_t | 16 | number of patches per chunk-AR round, e.g. 1/2/4/8/16/32; clamped to blockSize at construction |
| maxBatch | std::size_t | 4 | number of request slots per round |
| maxVariables | std::size_t | 8 | upper bound on the total of targets and covariates per request |
| cachePages | std::size_t | 2048 | upper bound on KV pages per channel count; one page stores one patch; allocated on demand; 0 disables the cache |
| maxSessions | std::size_t | 16 | number of sessions whose history is kept; the least recently used is evicted when exceeded; 0 creates no sessions |
| fullPrefillThreshold | std::size_t | 512 | when the session's retained history this round does not exceed this number of points, all new readings are used, allowing a re-prefill; 0 always fixes the boundary |
| threads | std::size_t | 1 | thread count for BLAS/Eigen/ytensor; default 1 serial, 0 automatic (at most 4), N for an exact thread count |
| continuousRope | bool | false | experimental continuous patch time positions |
| ropeReferenceInterval | double | 0.0 | reference sampling interval; 0 adopts the request interval |
| gpuThreads | std::size_t | 0 | Vulkan logical dispatch-window budget; 0 for unlimited; a positive value must be a multiple of 256; not a physical CU quota |
| attentionMask | std::function<bool(std::size_t, std::size_t)> | empty | additional visibility predicate on the time axis, arguments (query, key); empty is the native mask |

**Note**

The arguments of attentionMask are time-axis token indices: one token is patchSize sample points, counted from the earliest patch of the model context grid, with forecast patches continuing the count. A (query, key) pair returning false is masked in addition to the native causal mask; a token is always visible to itself. The predicate is called concurrently and must be reentrant. Only the reference and BLAS backends execute it natively; on the Eigen, ytensor, and Vulkan backends, setting it switches the computation to the reference model, with a noticeable speed drop. Because attentionMask is a std::function, Settings is still an aggregate, but it can no longer be used in constexpr.

### Timestral::Request

One forecast request. When the tails are not aligned, it is segmented and completed by observation availability; the horizon is counted from the latest valid observation; all channels share one time axis, and both the conditioning and the target are determined by the observation positions of variables itself. When session is empty, variables is the complete history; when non-empty, variables holds only the new readings after each channel's tail in that session, and may be empty.

**Members**
| Member | Type | Default | Description |
| --- | --- | --- | --- |
| session | std::string | empty string | the session identifier to continue, taken from the previous Result::session |
| streamId | std::string | empty string | the source identifier; an empty string is an anonymous request; continuing a session deduplicates by session |
| variables | std::vector<std::vector<Sample>> | - | each channel's time-increasing readings; when continuing, the channel count must match the session |
| horizonSeconds | double | 0.0 | forecast span in seconds; 0 takes patchBatch patches (5M default 512 points); negative is illegal |
| sampleInterval | double | 0.0 | sampling interval in seconds; 0 means infer from the timestamps |
| symmetricAveraging | bool | true | additionally do a mirror forward with negated context and average |
| clipNonnegative | bool | true | clip trajectories whose observations are all non-negative to be non-negative |

Sessions of synchronized channels use all new readings by default in the short-history phase: when the retained history of this round after trimming to maxContext does not exceed fullPrefillThreshold, a prefill is allowed to be redone due to patch realignment, with bufferedSamples of 0. The threshold defaults to 512 sample points, the same as the default one-round decode length of 5M, but it does not change with patchBatch or the current horizon. After exceeding the threshold, the patch boundary and left padding from the last time readings were fully accepted are retained, and when continuing only patches that are filled are sent to the model; the tail readings of an unfilled patch are saved in the session and do not participate in this round's forecast. The next append first concatenates with the tail, and complete old pages are not re-encoded because of a change in the remainder of the appended point count. This input rule does not depend on whether the KV cache is enabled; setting the threshold to 0 always fixes the boundary. The first request still uses all readings, keeping the upstream left-padding semantics.

During buffering, the internal forecast length is the required step count plus the buffered point count, and the corresponding leading span is then skipped. Result::origin is therefore still the time of the latest reading, and the returned step count and horizon are unchanged, but the forecast did not use the buffered readings and cannot be regarded as equivalent to a stateless request with all the latest observations. Result::bufferedSamples gives the number of buffered tail readings per channel, in the range 0 to patchSize minus 1. In the 5M default configuration, for example, when 31 points are appended after the history reaches 512 points, 16 pages are reused and 31 points are buffered, and the next append of 1 point adds only 1 page; appending 31 points at 160 points is still in the short-history phase, and all 191 points participate in the forecast.

After exceeding maxContext, the window moves in whole-patch steps, the history normalization and context change, and the window KV still needs to be rebuilt; no frozen-KV approximate sliding window is enabled here. The conditional completion of desynchronized channels still takes the staged full path and does not apply the tail buffering rule; the session saves the history the auxiliary channels need to cover the earliest position to be forecast plus the known future, not truncated independently by each channel's tail. When all latest readings must be used each time, submit the complete history without a session.

### Timestral::Result

The output of one forecast; time in seconds; the quantiles are stored in (channel, step, level) row-major order.

**Members**
| Member | Type | Description |
| --- | --- | --- |
| channels | std::size_t | number of channels, in the same order as the input |
| sampleInterval | double | the effective sampling interval |
| origin | double | the time step immediately before the forecast window |
| levels | std::vector<double> | the model's quantile points |
| quantiles | std::vector<double> | (channels, steps, levels) row-major |
| observed | std::vector<std::uint8_t> | (channels, steps), 1 for original observations |
| prefillSeconds | double | time spent encoding prefix pages |
| decodeSeconds | double | decoding time |
| totalSeconds | double | total execution time |
| reusedPages | std::size_t | number of hit prefix pages, excluding the pages of the symmetric-averaging mirror pass |
| computedPages | std::size_t | number of newly computed pages |
| bufferedSamples | std::size_t | number of tail readings buffered per channel that have not yet participated in this round's forecast, less than patchSize |
| session | std::string | the session saving this complete history; empty when maxSessions is 0 |

#### Result::steps()

The number of forecast window steps per channel.

**Returns**
| Type | Description |
| --- | --- |
| std::size_t | number of steps |

#### Result::time()

The timestamp of step.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | the step index from 0; must be less than steps() |

**Returns**
| Type | Description |
| --- | --- |
| double | origin plus (step+1) sampling intervals, in seconds |

**Throws**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step is not less than steps() |

#### Result::median()

The median of step, i.e. the 0.5 quantile.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | the step index from 0; must be less than steps() |
| channel | std::size_t | 0 | the channel index |

**Returns**
| Type | Description |
| --- | --- |
| double | the raw-scale median of that step |

**Throws**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step or channel out of range |

#### Result::quantile()

The quantile of step at any level.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | the step index from 0; must be less than steps() |
| level | double | - | the target level in (0,1); the grid endpoint is taken when outside the model quantile grid |
| channel | std::size_t | 0 | the channel index |

**Returns**
| Type | Description |
| --- | --- |
| double | the quantile of that step at level, in the raw scale |

**Throws**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step or channel out of range, or level not in (0,1) |

**Example**
~~~cpp
// 0.10 and 0.90 quantiles
double p10 = result.quantile(t, 0.10);
double p90 = result.quantile(t, 0.90);
~~~

#### Result::lower()

The lower bound of the central interval.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | the step index from 0; must be less than steps() |
| level | double | - | interval coverage in (0,1); level=0.9 takes the 0.05 quantile |
| channel | std::size_t | 0 | the channel index |

**Returns**
| Type | Description |
| --- | --- |
| double | the lower bound, in the raw scale |

**Throws**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step or channel out of range, or level not in (0,1) |

#### Result::upper()

The upper bound of the central interval.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | the step index from 0; must be less than steps() |
| level | double | - | interval coverage in (0,1); level=0.9 takes the 0.95 quantile |
| channel | std::size_t | 0 | the channel index |

**Returns**
| Type | Description |
| --- | --- |
| double | the upper bound, in the raw scale |

**Throws**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step or channel out of range, or level not in (0,1) |

#### Result::isObserved()

Whether that point is an original observation; a known position does not represent the model's prediction distribution.

**Parameters**
| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| step | std::size_t | - | the step index from 0 |
| channel | std::size_t | 0 | the channel index |

**Returns**
| Type | Description |
| --- | --- |
| bool | true for an original observation, false for a forecast |

**Throws**
| Exception | Condition |
| --- | --- |
| std::invalid_argument | step or channel out of range |

#### Result::cacheHitRate()

The paged-cache hit rate.

**Returns**
| Type | Description |
| --- | --- |
| double | reusedPages/(reusedPages+computedPages); returns 0 when there are no page statistics |

### Timestral::RequestSuperseded

When a request is invalidated by a newer same-source request or abort under the Mailbox policy, its future throws this exception.

**Inherits**

std::runtime_error
