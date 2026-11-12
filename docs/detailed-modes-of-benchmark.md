# Benchmark modes and options

Real-time replay (`kvlab benchmark auto --router-replay-file ...`) is the primary
way to benchmark with kvlab; see the [README](../README.md#real-time-replay) and
[replay.md](replay.md). This page lists the ways to vary that run, and the
alternative modes. Defaults below are those of the binary; run
`kvlab benchmark auto --help` for the authoritative text.

Contents: [1. Varying a real-time replay](#1-varying-a-real-time-replay) |
[2. Fixed-pool replay](#2-fixed-pool-replay) |
[3. Synthetic and dataset modes](#3-synthetic-and-dataset-modes) |
[4. Throughput command](#4-throughput-command) |
[5. Eval commands](#5-eval-commands) |
[6. Report commands](#6-report-commands) |
[7. Helm values](#7-helm-values)

Command tree: `benchmark` (`auto`, `throughput`, `visualize`, `visualize-merge`,
`timeline`) and `eval` (`simple-tool`, `coherency`). Global options (all
commands): `--stderr-logs`, `--log-format text|json`, `--log-http-requests`,
`--json-output`, `--tmp-dir`.

## 1. Varying a real-time replay

All flags are on `benchmark auto`. "RT" = applies to real-time replay (the
default with `--router-replay-file`). Under RT these defaults resolve: start 32
sessions, +16 every 1m, no session cap, `--ttft-retire` on, decode-rate floor 20,
timeline on.

### Model and replay file

| Flag | Meaning |
|---|---|
| `--models` / `--model` | Endpoint(s) to benchmark; repeatable, run concurrently. A bare `http(s)://` URL autodiscovers `/v1` and the model id (see the README). Helm: `endpoint`, `model`. |
| `--router-replay-file` | The replay file. Mutually exclusive with `--from-dataset`. Helm: `replay.replayFile`. |
| `--router-replay-roles` | Comma-separated instance roles to replay (default: all), e.g. `main,sub-agent` to drop helper side-calls. Helm: `replay.replayRoles`. |
| `--timeout` | Wall-clock limit of the run, prints the summary at the end. Helm: `duration`. |

### Admission and ramp (RT)

Mechanics are in [replay.md](replay.md#session-admission).

| Flag | Default | Meaning |
|---|---|---|
| `--start-series` | 32 | Sessions admitted at the start. Helm: `replay.startSessions`. |
| `--admit-every` | 1m | Tick interval. Helm: `replay.admitEvery`. |
| `--admit-count` | 16 | Sessions added per tick (bounded by `--max-series`). Helm: `replay.admitCount`. |
| `--max-series` | uncapped | Cap on active sessions. Helm: `replay.maxSessions`. |
| `--ttft-limit` | 10s | Admission pauses while the gate statistic is at or above it. Defines what "sustained" means for the run. Helm: `replay.ttftLimit`. |
| `--ttft-window` | 30s | Trailing window the TTFT statistic is taken over. Helm: `replay.ttftWindow`. |
| `--ttft-limit-stat` | mean | `mean`, `p50`, `p95` or `p99`. Helm: `replay.ttftLimitStat`. |
| `--ttft-retire` | true | Let the session count shrink: a finished session is not replaced while the gate is over. |
| `--ttft-retire-margin` | 0.15 | Fraction above `--ttft-limit` at which a finishing session is retired. |
| `--decode-rate-min` | 20 (RT) | Per-request decode tokens/s floor, a second gate; `0` = off. Helm: `replay.decodeRateMin`. |
| `--decode-rate-window` | 30s | Trailing window for the decode rate. |
| `--replay-allow-underfill` | false | Continue when the corpus is exhausted and slots cannot be filled (default: abort). Helm: `replay.allowUnderfill`. |

### Timing and traffic shape

| Flag | Default | Meaning |
|---|---|---|
| `--replay-timing` | `think` if the file has `think_ms`, else `absolute` | How a turn's due time is set. See [replay.md](replay.md#think-time). |
| `--replay-skip-idle` | false | Jump the clock when nothing is in flight and every session is waiting on a turn. Helm: `replay.skipIdle`. |
| `--no-stream` | false | Non-streaming requests; TTFT becomes whole-response time and the decode-rate gate is off. |
| `--max-output-tokens` | 0 (file budgets) | Cap each request's `max_tokens`. Helm: `replay.maxOutputTokens`. |
| `--replay-output-ratio` | 0 (off) | Retarget `max_tokens` to input tokens times this ratio instead of the recorded output tokens. |
| `--replay-chars-per-token` | 0 (off) | Size synthesized content as block tokens times this many chars, so the serving tokenizer lands near the original token counts. |
| `--limit-context` | 0 (none) | Skip requests whose recorded prompt tokens exceed this, to avoid 400 storms on small-context models. |
| `--replay-max-requests-per-session` | 0 (off) | Replay only the first N requests of each session, chronologically across the whole session (can remove a sub-agent, never orphan one). A capped run is not comparable with an uncapped one. Helm: `replay.maxRequestsPerSession`. |
| `--request-timeout` | 5m | Per-request timeout. Helm: `replay.requestTimeout`. |
| `--max-concurrency` | 0 | With `--concurrency` unset: upper bound for the concurrency hill-climber (see fixed-pool). Helm: `replay.maxConcurrency`. |
| `--concurrency` | unset | In-flight request gate. Unset under RT leaves it out of the way; setting it is a deliberate ceiling. Helm: `replay.concurrencyCap`. |

### Session choice

| Flag | Default | Meaning |
|---|---|---|
| `--replay-selection-strategy` | `reverse-from-end` | `reverse-from-end`, `beginning`, `last-week`, `random`. |
| `--replay-from` | unset | Start of the first lap: 0-based session index or RFC3339 time. Needs replay-v5. |
| `--replay-reuse-sessions` | true | Replay the corpus again from the top when it runs out; each lap gets its own content stamp. `=false` sweeps once. Helm: `replay.reuseSessions`. |
| `--replay-series` | 0 (all) | Cap on sessions replayed. Helm: `replay.replaySeries`. |
| `--replay-series-indices` | unset | Comma-separated 0-based session indices to replay (e.g. `3,7,42`). Overrides `--replay-series`. Exclusive with the range flag. |
| `--replay-series-range` | unset | Inclusive index range (`0-50`). Overrides `--replay-series`. |
| `--total` | 0 (unlimited) | Stop after this many requests. |

### Content

| Flag | Default | Meaning |
|---|---|---|
| `--replay-no-stamp` | false | Disable the per-run `<ignore>RUN_GUID</ignore>` stamp prepended to the system prompt (by default it makes a rerun start against a pristine prefix cache while within-run sharing still hits). Helm: `replay.replayNoStamp`. |
| `--seed` | 0 (random, printed) | Seed for the run identity: the run-id stamp and the `--verify` markers derive from it. Reusing a seed reproduces the content exactly, so the fleet may hold its prefixes. |
| `--cache-sim-chunk-bytes` | 0 (1024) | Chunk size of the content-level cache estimator. |

### Verification

Covered in the README ([Verify mode](../README.md#verify-mode-and-performance-mode)). Flags:
`--verify`, `--verify-continue-on-contamination`, `--verify-force-eos`,
`--dump-dir`, `--dump-limit` (200), `--dump-garbage` (true),
`--dump-garbage-dir`, `--dump-miss-dir`, `--dump-miss-substituted-only`.
`--verify` is router-replay only. Helm: `verify.enabled`, `verify.forceEos`,
`verify.continueOnContamination`.

### Run control and output

| Flag | Default | Meaning |
|---|---|---|
| `--save-request-data` | unset | Directory for per-request JSONL (+ `report.html`). The chart always sets it. |
| `--timeline` | on under RT | Embed the request-timeline view in the report; needs `--save-request-data`. |
| `--live-port`, `--live-bind`, `--live-max-requests` | off, `127.0.0.1`, 100000 | Live timeline server; see [replay.md](replay.md#results-kv-details-and-the-timeline). |
| `--vllm-metrics-url`, `--sglang-metrics-url` | derived from `--models` | Where to scrape the cache-source counters. Helm: `replay.vllmMetricsUrls`. |
| `--print-errors-threshold` | unset | Print errors to stderr, at most one per interval. Helm: `replay.printErrorsThreshold` (chart default `1s`). |
| `--max-consecutive-failures` | 512 | Abort after this many failures in a row. |
| `--max-total-errors` | 0 (off) | Abort after this many total errors; 1 stops at the first. |
| `--abort-on-collapse` | false | Abort if the windowed cache hit rate stays below 50% for 2 minutes. Helm: `replay.abortOnCollapse`. |
| `--replay-stop-at-low-concurrency` | false | Stop once the queue is drained and active workers fall below `--concurrency`. Helm: `replay.replayStopAtLowConcurrency`. |
| `--random-gate-order` | true | Wake waiting sessions on the concurrency gate in random order; `=false` is strict FIFO. |
| `--endpoint-overload-threshold` | 1.2 | Multi-endpoint specs (`a\|b\|c`): a series sticks to its endpoint until its in-flight exceeds this multiple of its fair share, then fails over. |
| `--dry-run` (+ `--dry-run-cold-tps` 1000000, `-warm-tps` 10000000, `-output-tps` 100000) | false | Drive the replay pipeline with synthetic timing and no HTTP requests, to watch cache evolution offline. Requires `--router-replay-file`. Helm: `replay.dryRun`, `replay.dryRun{Cold,Warm,Output}TPS`. |

## 2. Fixed-pool replay

`--replay-realtime=false` replays a fixed pool of sessions back-to-back with no
think time and no TTFT admission. It asks "how does the fleet behave at
concurrency N" rather than "what is the most sessions this fleet can carry".

```
kvlab benchmark auto \
  --router-replay-file replay.jsonl \
  --models http://YOUR-LLM-HOST:8000 \
  --replay-realtime=false \
  --series 256 --concurrency 28 --hot-series-concurrency 4 \
  --timeout 8h \
  --save-request-data ./results
```

- `--series N`: how many parallel workers replay sessions; each pulls the next
  session from the stream and runs its tree to completion. Pins both
  `--start-series` and `--max-series`; refused without `--replay-realtime=false`
  (it would pin the count real-time replay grows), and exclusive with those two.
- `--replay-series N`: a subset cap on which sessions are replayed at all (0 =
  every session). Not the same as `--series`.
- `--concurrency N`: gate on simultaneously in-flight HTTP requests; a session
  tree fanning out can want more slots than its one worker, and the gate queues
  the surplus. `0`/unset enables the concurrency hill-climber, bounded by
  `--max-concurrency` (0 = unlimited).
- `--hot-series-concurrency H`: H of the workers form a "hot" pool with a
  dedicated gate, issuing back-to-back requests while the rest share
  `--concurrency` (a foreground-agents plus background-fleet mix). 0 = off.
- Ramp: instead of `--series`, use `--start-series=32 --max-series=256
  --admit-count=32 --admit-every=2m` to add 32 sessions every two minutes; the
  first increase happens after two minutes. A batch is capped only when
  `--max-series` is set. Without `--admit-count` one series is added per tick.
  Outside real-time replay admission depends only on the interval, not on a gate.
- Interaction: with `--admit-every`, `--max-series` defaults to unlimited,
  otherwise 64; `--start-series` defaults to 1.
- `--decode-rate-min` is a real-time feature and is rejected above 0 here;
  `--replay-skip-idle` warns that it has nothing to skip.

Helm: `replay.realtime=false` selects it; then `replay.series` (256),
`replay.concurrency` (28; 0 = hill-climber), `replay.hotConcurrency` (4; 0 = off)
and `replay.maxConcurrency` are passed. Under the default `replay.realtime=true`
they are ignored (`replay.series`, `concurrency`, `hotConcurrency`), except
`replay.concurrencyCap`, which sets `--concurrency` as a deliberate ceiling.

## 3. Synthetic and dataset modes

`benchmark auto` without `--router-replay-file` generates its own load. These
modes answer "how does the endpoint scale on a controlled prompt shape" and need
no captured traffic.

**Embedded-documentation load test** (default when no replay file or dataset is
given). Prompts are built from an embedded document (or `--docs-dir`), each
series with a unique GUID so prefixes do not collide. The run auto-scales:
series grow while the cache hit rate holds.

```
kvlab benchmark auto --models http://YOUR-LLM-HOST:8000 --timeout 30m \
  --save-request-data ./results
```

| Flag | Default | Meaning |
|---|---|---|
| `--docs-dir` | embedded doc | Documentation directory to embed. |
| `--tokens` | 100000 | Upper bound on per-request prompt tokens (truncates the embedded doc, max 305465; with `--docs-dir` caps `--step` growth). |
| `--step` | off | Token step for incremental prefix growth per request (`3k`, `3000`), simulating agentic context growth. |
| `--step-starting-tokens` | `--step` | Initial prefix size per series when `--step` is on. |
| `--shared-prefix-per-series N` | 0 | Every N series share the same prefix (unique session id appended), testing cross-series reuse. Carries a per-run nonce so it cannot read another run's cache. |
| `--cache-target` | 0.90 | Cache hit rate at which series scaling begins. |
| `--cache-window-size` | 20 | Recent requests used to measure the hit rate. |
| `--ttft-hit-threshold` | 0.5 | Fraction of the cold-start TTFT below which a request counts as a hit. |
| `--ttft-degradation-factor` | 4 | TTFT multiple over the early cold baseline at which the heuristic is disqualified. |
| `--global-cache-hit-rate-target` | 0 (off) | Do not spawn new series until the all-time hit rate reaches this. |
| `--exhaust-sessions` | false | When a series' `--step` prefix reaches the token cap, recycle the slot with a fresh session. Requires `--step`. |
| `--profile agentic` | unset | Preset: step 3k, shared-prefix 4, hit-rate target 0.95, tokens 200000, max-output 32000, exhaust-sessions on (each only if not set explicitly). The only profile. |
| `--max-series`, `--start-series`, `--series`, `--concurrency`, `--max-concurrency`, `--admit-every`, `--admit-count` | see section 2 | Same scaling controls. |
| `--verbose-cache`, `--print-responses` | false | Diagnostics; both force non-TTY display. |

**Dataset replay** (`--from-dataset <short-name>`, e.g. `hermes-lambda`). Replays
conversations from a dataset instead of synthetic prompts: each series is one
conversation, and when done its slot pulls the next from the queue until the
queue drains. `--replay-series N` limits how many rows are loaded; with it (and
no `--series`/`--admit-every`) start and max series default to N. Defaults
`--max-output-tokens` to 32000. Per-run stamping (`--replay-no-stamp`),
`--limit-context` and `--replay-stop-at-low-concurrency` apply.

## 4. Throughput command

`kvlab benchmark throughput` measures three phases on the same prompts: cold
prefill, warm prefill and decode. It answers "what are this endpoint's raw
prefill and decode rates at concurrency C".

```
kvlab benchmark throughput --models http://YOUR-LLM-HOST:8000 \
  --starting-concurrency 16 --decode-tokens 1000
```

| Flag | Default | Meaning |
|---|---|---|
| `--models` | required | Models to benchmark; repeatable. |
| `--starting-concurrency` | 16 | Concurrency C for all phases (and the first sweep level). |
| `--cold-prefill-concurrency` | 0 | Separate concurrency for the cold phase only (0 = C). |
| `--series-per-concurrency` | 1 | Total series = C times this. |
| `--decode-tokens` | 1000 | `max_tokens` for the decode phase. |
| `--tokens` | 100000 | Truncate the embedded doc to this many tokens; `--docs-dir` uses your own. |
| `--auto-sweep` | false | Double concurrency from `--starting-concurrency` until decode rate plateaus. |
| `--max-concurrency` | 64 | Upper bound of the sweep. |
| `--sweep-improvement-threshold` | 0.05 | Stop when decode-rate improvement falls below this fraction. |
| `--high-series-check N` | 0 (off) | After the main run, measure warm prefill at N series. |
| `--timeout`, `--question` | | Run limit; question text. |

## 5. Eval commands

### `kvlab eval coherency`

Verifies that an inference server's prefix/KV cache returns correct bytes, not
just fast ones. Each series gets a large garbage-padded system prompt (213,000
characters by default) with a list of unique UUID stamps scattered through it;
the model's only job is to echo its own comma-joined UUID list. Any deviation is
evidence of cache corruption, not model weakness.

```
kvlab eval coherency \
  --model dynamic/http://YOUR-LLM-HOST:8000,type=openai_vllm \
  --series 256 --shared-prefix-per-series 4 --abort-fraction 0.1
```

**Cycles.** With `--total` unset the run sends `2 x series` requests: cycle 1
sends each unique prompt cold, cycle 2 re-sends the identical prompts so they
should be served from cache. The report prints mean cold TTFT (cycle 1) against
mean warm TTFT (cycle 2) and an implicit hit rate (a cycle-2 request counts as a
hit when its TTFT is at most 50% of the cycle-1 mean).

**Report classes.** `UUID_MISSING_FLAKY`: an expected UUID is absent (truncated
or corrupted prefill, or a block served from the wrong content).
`CROSS_CONTAMINATION`: a UUID of a different series appears, a KV or scheduling
leak and the worst class. `NOT_EXACT`: all UUIDs correct but extra prose or
whitespace. `ERROR`: the request failed.

**Shared cache.** `--shared-prefix-per-series N` groups series into cohorts of N
sharing one byte-identical leading garbage prefix, so peers co-hit the same
prefix-cache blocks; each series' UUID stamps still trail it, so contamination
detection stays per-series. Default 0: every series fully unique. The shared
block carries a per-run nonce, so a run reads only KV it wrote itself.

**Adversarial aborts.** `--abort-fraction F` cancels that fraction of requests
mid-flight (connection closed during prefill or KV load). Aborted requests are
excluded from scoring; only their count is reported. `--abort-delay-ms` fixes
the delay (default 0: random within the live cold-TTFT estimate).

Other flags: `--series` (1), `--concurrency` (1; raise it to create contention),
`--total`, `--seed` (0 = random), `--garbage-characters` (213000; the older
`--garbage-tokens` maps to characters at 4 per token), `--reset-every-n` (POST
vLLM's dev-mode `reset_prefix_cache` every N completed requests; needs
`VLLM_SERVER_DEV_MODE=1`), `--max-output-multiplier` (3.0; response `max_tokens`
budget for reasoning models), `--full-missing-responses` (untruncated failure
dumps). Alias: `cache-coherency-garbage-clean`.

### `kvlab eval simple-tool`

Evaluates tool-calling: the model must drive an increment tool through a chain of
calls. `--model` (required, any model spec) and `--target` (default 10, number of
tool calls in the chain).

## 6. Report commands

- `benchmark visualize <dir>`: `report.html` from a results directory.
  `--timeline`, `--overwrite`, `--max-elapsed` (drop records past an elapsed
  time, e.g. to strip a crash's error storm), `--concurrency` (smoothing window
  override).
- `benchmark visualize-merge <dirs...>`: combine runs into one report. `--all`
  (treat the argument as a parent directory), `-o`, `--labels`, `--baseline`
  (arm the % comparison is relative to), plus `--timeline`, `--overwrite`,
  `--max-elapsed`, `--concurrency`.
- `benchmark timeline <dir> -o timeline.html`: just the timeline view.

## 7. Helm values

Run-shape values in `chart/kvlab/values.yaml`, grouped by what they select. Values
left empty are not passed, so the binary's defaults apply.

| Group | Values |
|---|---|
| Mode | `replay.realtime` (true; `false` = fixed pool) |
| Real-time ramp and gates | `replay.startSessions`, `maxSessions`, `admitEvery`, `admitCount`, `ttftLimit`, `ttftWindow`, `ttftLimitStat`, `decodeRateMin`, `skipIdle`, `allowUnderfill`, `reuseSessions` (true) |
| Fixed pool only | `replay.series` (256), `concurrency` (28), `hotConcurrency` (4) |
| Both | `replay.maxConcurrency`, `concurrencyCap` (real-time ceiling), `requestTimeout`, `maxOutputTokens`, `maxRequestsPerSession`, `replaySeries`, `replayRoles`, `replayNoStamp` |
| Run control | `duration` (8h), `abortOnCollapse`, `replayStopAtLowConcurrency`, `printErrorsThreshold` (`1s`), `stderrLogs`, `vllmMetricsUrls` |
| Offline simulation | `replay.dryRun`, `dryRunColdTPS`, `dryRunWarmTPS`, `dryRunOutputTPS` |
| Verification | `verify.enabled`, `verify.forceEos`, `verify.continueOnContamination` |
