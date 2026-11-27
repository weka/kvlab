# Real-time replay in depth

Reference for `kvlab benchmark auto --router-replay-file`, the primary run mode
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

`kvlab router serve --capture redacted` records live traffic;
`kvlab router replay-prepare` compiles those captures into one replay-v5 JSONL
file (replay-v4 and v3 still load): a header line, one session per line in
start-time order, and a final index line, so the replayer streams it with a
bounded queue regardless of file size. Each request embeds the original
structured spec (system blocks, tools, messages) with content regenerated
deterministically from each block's hash: same hash, same bytes, so the server's
prefix cache hits the way the original traffic did, without shipping any real
captured text.

The capture keeps one byte count per system block, per tools array and per
message (the canonical JSON of each), not per content block. Replay sends each
of them at that size: a message's bytes, less the JSON structure of the message
it rebuilds (roles, block types, tool ids and names), are split over all its
content blocks by block-type weight: text 1000, `tool_use` input (a `_padding`
string) 1736, `tool_result` content 2364, thinking 6394, anything else (images,
documents, sent as text of their share) as text. The weights are fitted on raw
captures that still had content, so each type's bytes summed over those
messages match the real ones (`tool_use` 0.99, `tool_result` 1.05, text 1.00,
thinking 0.99). A thinking block's share is 12% thinking text and 88%
`signature`, as measured; a server does not tokenize the signature, and the
OpenAI format has no place for it, so it is not sent there. The tools array's
bytes go to the tool descriptions evenly. Both wire formats carry the same
content otherwise. Token counts then follow from the
serving tokenizer's density on the synthesized text (`--replay-chars-per-token`
sizes from captured tokens instead).

## Token calibration

Synthesized text tokenizes at the serving tokenizer's own density, not the
original traffic's, so byte-faithful content carries only part of the captured
prompt tokens (roughly 45-75% depending on the model). Before a replay starts,
kvlab calibrates a content scale against the running server:

1. **Probe**: the first request the run would send (role, no-usage and
   `--limit-context` filters applied, `--replay-drop-late-system` trimmed) whose
   captured prompt is over 10,000 tokens. It is built at scale 1, sent with
   `max_tokens` 1 and its own content stamp (a `calib-` nonce, so its KV is no
   run request's prefix), and recorded nowhere in the results.
2. **Adjust**: the server's prompt tokens against the captured ones give the
   next scale (a multiplicative step, then secant on the last two probes).
3. **Accept** the first scale whose probe lands within 3% of the captured
   prompt, after at most four adjustments. A run that cannot get there does not
   start.

The scale multiplies the bytes of everything the server tokenizes: system text,
tool descriptions (the same synthesized text, so the same density), and message
content including `tool_use` input. JSON structure and thinking signatures keep
their captured sizes, so the summary's input **bytes** conformity sits above
100% by about the scale while input **token** conformity sits near 100%. The
OpenAI, sglang and Anthropic paths read the probe's prompt tokens the same way
they read every response's usage. `--limit-context` still compares captured
tokens.

The value is printed at start and in the summary beside the seed:

```
Replay sizing [model]: seed=8231 replay-token-scale=1.4375 (calibrated)
  reproduce these exact bytes with --seed=8231 --replay-token-scale=1.4375 (--seed alone reproduces content, not sizes)
```

and recorded in the run params (`seed`, `replay_token_scale`,
`replay_token_scale_source`), which the HTML report shows. **`--seed` alone no
longer reproduces a calibrated replay's bytes**: pass both. An explicit
`--replay-token-scale` (1 = byte-faithful) or `--replay-chars-per-token` skips
calibration; dry runs do not calibrate.

Calibration is setup, not load: every model calibrates (concurrently, each
against its own server) before any model's run starts, and **`--timeout`
counts from after calibration**, so a 30-minute run measures 30 minutes of load
however long the probes took. Elapsed time, the run-params `ts`, the live view
and the realtime session clocks all start there too. Calibration has its own
bound instead: 20 minutes per model (each probe at most 5); past it the run
fails with `calibration exceeded its 20m0s bound` rather than starting. The log
shows the split:

```
[calibrate][model] sizing replay content to the server's tokenizer (bound 20m0s; not counted in --timeout)...
[calibrate][model] probe session ... scale 1.0000 -> 6116 (59.2%); scale 1.6895 -> 10235 (99.1%)
[calibrate][model] done in 41.3s (setup, outside --timeout)
[auto] load starts now; --timeout=30m0s counts from here
```

and the run params record it as `calibration_sec`.

## Volume conformity

Every replay run reports how much of the captured volume it put through the
server, in the summary

```
 Volume conformity  : input 73.4% (bytes 100.3%), output 91.0%
                      input 1174.9M of 1600.6M captured tokens, 4914.2M of 4899.5M captured bytes; output 3.1M of 3.4M requested
```

and as `vol=in:..,bytes:..,out:..` on the progress line:

- **input**: prompt tokens the server reported (uncached + cached) over the
  capture's prompt tokens for the same requests. The capture's `input_tokens`
  is the whole prompt (prefill + cache read + cache creation). Near 100% when
  the scale is calibrated; at scale 1 it is the serving tokenizer's density on
  synthesized text relative to the original traffic.
- **bytes**: the request body as posted over the capture's request bytes
  (system blocks + tools + messages). About 100% at scale 1, and above it by
  about the content scale when calibrated (see
  [Token calibration](#token-calibration)); slightly above from the run
  stamp, output instruction and length ask; below when
  `--replay-drop-late-system` leaves messages out, and on the OpenAI format,
  whose message structure is lighter than the capture's Anthropic one and
  which drops thinking signatures.
- **output**: completion tokens over the `max_tokens` each request carried;
  ~100% under `ignore_eos`, ~90% under `--verify`.

Each is a sum over the requests that completed without error. The request data
carries the operands and per-request ratios: `volume_captured_input_tokens`,
`volume_captured_input_bytes`, `volume_wire_input_bytes`,
`volume_output_target_tokens`, `volume_input`, `volume_input_bytes`,
`volume_output`. The HTML report's summary panel has a **Volume** column
(bytes on hover) for the selected timeframe, and its CSV exports carry the
operand columns.

A session instance whose prompt outgrows the context is retired: its remaining
turns are not sent, since prompts only grow. `--limit-context` retires at the
first request whose captured prompt tokens (`input_tokens`, cached part
included) exceed the limit; a server context-length 400 retires too. Both are
counted by reason on the progress line (`retired=limit:N,overflow:M`), in the
summary (`Retired (context)`), as `instance_retired` rows in the request data,
and on the arm's row in the HTML report.

`replay-prepare --since/--until` bound the capture by request time and
`--complete-by` keeps only requests whose response had finished by an instant,
for rebuilding a corpus from a directory that has grown.
`kvlab router replay-trim --last-days N` (or `--last-sessions N`) cuts a smaller
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
(TTFT / response time / cache hits over time). `kvlab benchmark visualize <dir>`
regenerates it and `kvlab benchmark visualize-merge` combines runs. The
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

**Timeline.** `kvlab benchmark auto --timeline` (on by default under real-time
replay), `visualize --timeline`, `visualize-merge --timeline` and
`kvlab benchmark timeline <dir> -o timeline.html` add a Metrics/Timeline switch.
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
`record_type: "cache_metrics_sample"`, fields `ts`, `model`, `engine`, `bands`
(`compute` / `host` / `external`; `local_cache_hit` is reported as `host`), plus
the locally computed `active_dataset_tokens` and `active_series`. Older
`vllm_metrics_sample` / `sglang_metrics_sample` rows are still read. Point the sampler
elsewhere with `--vllm-metrics-url` (repeatable; e.g. the router's metrics
listener or every backend; totals are summed as per-endpoint deltas). Helm:
`replay.vllmMetricsUrls`.

**SGLang** (`type=openai_sglang`). The sampler reads `sglang:cached_tokens_total`
(`cache_source` = `device` / `host` / `storage`, plus an untiered `total`
fallback) and `sglang:prompt_tokens_total` and writes `cache_metrics_sample`
rows with `bands.{compute,gpu,host,storage}` (`device` is reported as `gpu`), where
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
