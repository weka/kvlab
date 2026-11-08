# Real-time replay in depth

Reference for `wekai benchmark auto --router-replay-file`, the primary run mode
(summarised in the [README](../README.md#real-time-replay)). For the other flags
and run modes see [detailed-modes-of-benchmark.md](detailed-modes-of-benchmark.md).

## Session admission

Real-time replay grows the number of active sessions while the fleet keeps up:
it starts `--start-series` sessions (default 32) and adds `--admit-count`
(default 16) every `--admit-every` (default 1m). The unit is the session:
admission starts sessions, active sessions are those started and not yet ended,
and the status line reports
`sessions=<active>(started=,ended=,unreplaced=,corpus=)`.

Two gates decide whether a tick may add sessions:

- **TTFT gate.** The windowed TTFT statistic over `--ttft-window` (30s) must be
  under `--ttft-limit` (10s). `--ttft-limit-stat` picks the statistic: `mean`
  (default), `p50`, `p95` or `p99`. The mean is tail-driven and closes at about
  half the session count a median gate would; all four are reported in the
  progress line whichever one gates. A duration window (not a request count) keeps
  the gate comparable as the fleet fills.
- **Decode-rate gate.** `--decode-rate-min` (default 20 tokens/s per request;
  `0` = off) over `--decode-rate-window` (30s). The rate is output tokens/s of
  requests in their decode phase, from the first token on, whatever their
  outcome, using per-chunk counts from the engine's `continuous_usage_stats`.

Requests are always sent when due; nothing is held back by the gates. An empty
TTFT window does not open admission while a due request has waited past
`--ttft-limit` for its first token.

A batch is admitted only when both gates stayed open for the whole
`--admit-every` interval before it (sampled every 5s), not just at the tick.
Once either gate has blocked, admission resumes only after three such open
intervals in a row: growth is fast until the first block and cautious after it.
Admission paused by a closed gate resumes with one batch per tick, without
catching up on skipped ticks.

**Retirement.** `--ttft-retire` (default on) lets the count shrink too. When a
session ends while the gate statistic is more than `--ttft-retire-margin` (0.15)
above the limit, or the decode rate is below the floor, no new session replaces
it. Between the limit and limit+margin sessions are replaced but none are added;
below the limit admission adds batches (up to `--max-series` active sessions).

**Caps and end of corpus.** `--max-series` caps active sessions (unset: no cap
under real-time replay). `--replay-reuse-sessions` (default true) replays the
corpus again from the top when it runs out, so a run lasts until `--timeout`;
each lap gets its own content stamp. `--replay-allow-underfill` lets a run
continue when the corpus is exhausted and admitted slots cannot be filled; by
default the run aborts, so a decaying run is not reported as a steady one.

**Admission rate is not neutral.** On a fleet that answers fast, the ramp rate
rather than the fleet can become the limit. Run two rates and check the answer
stopped moving before reading a plateau as capacity.

`--no-stream` sends every request non-streaming. TTFT then equals the whole
response time, so `--ttft-limit` gates end-to-end latency (size it accordingly),
and the decode-rate gate needs streamed usage, so the unset `--decode-rate-min`
resolves to off and an explicit value above 0 is rejected.

## Where the replay file comes from

`wekai router serve --capture redacted` records live traffic;
`wekai router replay-prepare` compiles those captures into one replay-v5 JSONL
file (replay-v4 and v3 still load): a header line, one session per line in
start-time order, and a final index line, so the replayer streams it with a
bounded queue regardless of file size. Each request embeds the original
structured spec (system blocks, tools, messages) with content regenerated
deterministically from each block's hash: same hash, same bytes, so the server's
prefix cache hits the way the original traffic did, without shipping any real
captured text.

`replay-prepare --since/--until` bound the capture by request time and
`--complete-by` keeps only requests whose response had finished by an instant,
for rebuilding a corpus from a directory that has grown.
`wekai router replay-trim --last-days N` (or `--last-sessions N`) cuts a smaller
replay out of a replay-v5 file: it keeps sessions that started in the window and
excludes sessions whose last request falls in the final `--incomplete-window`
(default 24h) of the capture.

## Think time

Each request carries `total_ms` and `think_ms`: the gap between the END of its
reference (`think_ref`: `prev` = the same agent's previous request, `spawn` = the
parent request that spawned a sub-agent, `session_start`) and its own start,
clamped at 0.

`--replay-timing` sets how a turn's due time is computed under real-time replay:

- `think` (default when the file carries `think_ms`): due at the end of the
  reference on THIS run's clock plus the captured think time, so a fleet that
  answers faster or slower than the capture keeps the user's pauses.
- `absolute`: the turn's captured offset from the session start, so a slower
  fleet falls behind.

A request without `think_ms` always uses `absolute`. `--replay-timing=think`
against a file with no `think_ms` is an error. `--replay-skip-idle` jumps the
clock when nothing is in flight and every active session is waiting on its next
turn, so a run does not spend its budget on dead time; it is safe because an
idle fleet has no in-flight request to distort, and rarely fires under load.

## Session index and selection

The last line of a replay-v5 file is `{"_schema":"replay-index-v1","sessions":[...]}`:
per session its byte `offset` and `length`, `start_ts`, `end_ts` (when its last
request finished) and `requests`. The replayer finds it by reading the file's
last 2 MiB (doubling until the window holds the whole line) and then reads each
session at its offset, which lets a run choose its sessions instead of reading
front to back. `--replay-selection-strategy`:

- `reverse-from-end` (default): whole corpus newest first every lap, so a run
  opens on the most recent traffic and, needing more sessions than the corpus
  holds, starts again from the end.
- `last-week`: the first lap plays the capture's final seven days in file order,
  then every lap is the whole corpus in file order.
- `beginning`: whole corpus in file order every lap.
- `random`: a fresh (per-lap seeded) permutation of the whole corpus every lap.

`--replay-from=<index|RFC3339>` moves where the first lap of `beginning`,
`reverse-from-end` or `last-week` starts (not valid with `random`). An unindexed
v3/v4 file plays from the beginning under the default; every other strategy
needs v5. Every lap gets its own content stamp.

## Results, KV details and the timeline

`--save-request-data <dir>` writes per-request JSONL under
`<dir>/<run-timestamp>/` and, at the end of the run, an interactive `report.html`
(TTFT / response time / cache hits over time). `wekai benchmark visualize <dir>`
regenerates it and `wekai benchmark visualize-merge` combines runs. The
**output/s/worker** toggle adds a line: output tokens from requests completed in
the trailing 60 seconds, divided by the window and the configured concurrency
(including idle slots); each arm uses its recorded concurrency, falling back to
the visualization's `--concurrency`, and unknown concurrency draws no line.

**Per-request KV details.** Each reqdata row records, when the engine reports
them, the prompt's token sources (`kv.sources`: gpu, host, external, computed),
recompute counters, and engine-side timings in seconds (`kv.timings`:
queue_wait_kv, queue_wait_other, kv_fetch, prefill, decode, preempted_time). vLLM
supplies them with `--enable-request-kv-details`
(`usage.prompt_tokens_details.kv_sources`, `usage.vllm_request_timings`);
SGLang's tier split is mapped onto the same sources. Nothing is recorded for
engines that report neither. Streaming requests already ask for
`stream_options.include_usage`, which is how vLLM delivers them.

**Timeline.** `wekai benchmark auto --timeline` (on by default under real-time
replay), `visualize --timeline`, `visualize-merge --timeline` and
`wekai benchmark timeline <dir> -o timeline.html` add a Metrics/Timeline switch.
Router replay draws one row per SESSION from its first request to its last
(labelled `#<run>`, in start order; a corpus session replayed on a later lap is a
new run), with overlapping requests (sub-agent fan-out) packed into sub-rows.
Reqdata records `session_run` per request; files that recorded only `slot`
(fixed-pool modes) get the same layout, each stretch of a worker's requests under
one `series_num` being one session; files with neither group by `series_num`.
x is time, rows are ordered by first request.

Each request bar, left to right: waits, by where they happened (client wait: the
turn was due but no concurrency slot was free, `ready_ms` to `sent_ms`; engine
queue; unattributed network/frontend), then prefill split by token share into
computed / external / host / GPU cache (kv_fetch counted in the external share),
preempted, decode. Requests with no engine details use client timestamps and a
single "prefill (unknown source)" colour. Time backing off after a 429 is its own
wait (`retries_429`, `retry_wait_ms`). A session's think pause is shaded faintly
(`think_pause_ms`).

Under the toolbar a state line counts active sessions as in flight / waiting at
gate / think pause / idle and, live, the admission gate (statistic over window vs
limit, open or closed, next admit). Strips above the rows show token throughput
(input stacked by source, output as a line) and the gate's TTFT (mean / p95
against the limit); the header shows tokens/s over 10 s and 60 s. Prompt tokens
count when prefill ends and output tokens spread over the decode.

**Live view.** `--live-port 8088` serves the timeline while the run executes
(polled every 2s, in-flight requests growing). It binds `127.0.0.1` (reach it with
`kubectl port-forward pod/<pod> 8088:8088`); `--live-bind 0.0.0.0` exposes
request metadata to the network. `--live-max-requests` (default 100000) bounds
memory per arm. `/api/state` serves the counts, `/api/series` the strips' history
and `/api/data?arm=` the arm by index or name.

## Engine metrics sampling

With `--save-request-data` on and an OpenAI-compatible (chat/completions) model
spec, a sampler polls the server's Prometheus endpoint (`<base>/metrics`, `/v1`
stripped) once a minute and appends the result to the same JSONL; this splits the
report's prompt tokens into compute vs local cache vs external cache.

**vLLM.** One series is read, `vllm:prompt_tokens_by_source_total`, keeping the
`source` values `local_compute`, `local_cache_hit` and `external_kv_transfer`,
summed over every other label. Each sample is a row with
`record_type: "vllm_metrics_sample"`, fields `ts`, `model`,
`sources.compute` / `sources.local_cache` / `sources.external_cache`, plus the
locally computed `active_dataset_tokens` and `active_series`. Point the sampler
elsewhere with `--vllm-metrics-url` (repeatable; e.g. the router's metrics
listener or every backend; totals are summed as per-endpoint deltas). Helm:
`replay.vllmMetricsUrls`.

**SGLang** (`type=openai_sglang`). The sampler reads `sglang:cached_tokens_total`
(`cache_source` = `device` / `host` / `storage`, plus an untiered `total`
fallback) and `sglang:prompt_tokens_total` and writes `sglang_metrics_sample`
rows with `sources.{device,host,storage,compute}`, where
`compute = max(0, prompt - (device + host + storage + total))`.
`--sglang-metrics-url` is the counterpart of `--vllm-metrics-url`. SGLang requests
also carry `return_cached_tokens_details`; the per-request tier split is stored as
`cached_device` / `cached_host` / `cached_storage` (+ `cached_storage_backend`) and
the run summary ends with a `sglang cache tiers:` line.

Collection is best-effort and never affects the benchmark: an unreachable or
non-Prometheus endpoint just skips samples. A spec that did not say
`type=openai_vllm` gives up after 3 consecutive failures rather than polling a
public API for the whole run; spelling out `type=openai_vllm` asserts the server
is vLLM and keeps it polling through restarts and slow weight loads.
