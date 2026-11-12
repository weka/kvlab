# kvlab (formerly wekai)

A real-time replay benchmark, model-aware LLM router, and capture toolkit for
OpenAI (Chat Completions), OpenAI Responses API, Anthropic and Google protocols.

## Philosophy

- **Replay:** the most realistic replay of captured agentic traffic, with minimal
  configuration.
- **Router:** zero parameters where possible, the easiest configuration with
  optimal performance.
- Where we find gaps in either approach, how requests are replayed or routed may
  change in a breaking way. **Compare runs only head to head on the same
  version**: a change in defaults or replay behaviour can move numbers on its own.

kvlab has three command groups: `benchmark` (replay and load tests, result
visualization), `router` (see below) and `eval` (model capability evaluations).
It works against any endpoint, given as a plain `http(s)://` URL.

## Install

```
go build -o kvlab .
# or: go install github.com/weka/kvlab@latest
```

```
kvlab benchmark auto --help
kvlab router serve --help
kvlab eval coherency --help
```

Released binaries, images and charts: see [Releases](#releases-and-license).

## Real-time replay

`kvlab benchmark auto --router-replay-file` replays previously captured agentic
traffic against an endpoint. It runs **in real time by default**, which is also
the Helm chart's default, and answers "what is the most sessions this fleet can
carry":

```
kvlab benchmark auto \
  --router-replay-file replay.jsonl \
  --models http://YOUR-LLM-HOST:8000 \
  --timeout 8h \
  --save-request-data ./results
```

That one command is the full recipe: captured think times between turns, start 32
sessions and add 16 every minute while the windowed mean TTFT is under
`--ttft-limit` (10s) and the per-request decode rate is at least 20 tok/s, no
session cap, retirement of sessions while a gate is closed, sessions reused from
the corpus so the run lasts until `--timeout`, and the request timeline in the
report. Always pass `--save-request-data`: it is where per-request JSONL and the
report go.

**What replay preserves.** It is not a request firehose. Each captured session is
re-enacted as a tree: per-session boundaries, sequential turn order per agent,
parent-to-child sequencing (a sub-agent starts only after its parent's spawning
request completes), sibling fan-out concurrency, and per-request input/output
token budgets. Prefix growth and sub-agent bursts hit the endpoint's prefix cache
with the same shape as real agentic load. Content is regenerated
deterministically from block hashes, so no real captured text is shipped, and each
run is stamped with a per-run `<ignore>RUN_GUID</ignore>` (`--replay-no-stamp`
disables it) so a rerun starts against a pristine prefix cache.

**How admission works.** Sessions are the unit. A batch is admitted only when the
TTFT gate (windowed mean TTFT below `--ttft-limit`) and the decode-rate gate
(`--decode-rate-min`, default 20 tok/s per request) both stayed open for the whole
`--admit-every` interval; after a block, admission resumes only after three open
intervals in a row. A session that ends while a gate is closed is not replaced
(`--ttft-retire`). Requests are always sent when due; nothing is held back. Gate
detail: [docs/replay.md](docs/replay.md#session-admission).

**Think time.** Replay really sleeps. Between turns it waits the captured think
time, measured from the end of the reference request on this run's clock, so a
faster or slower fleet keeps the user's pauses.

**Sessions change over time.** A session's requests are densest at its beginning,
then thin out into a tail that can run for hours or days. Real-time replay keeps
that shape, so the load a session puts on the fleet changes as it ages, and a
run's total load depends on its mix of young and old sessions.

**Where the replay file comes from.** `kvlab router serve --capture redacted`
records traffic and `kvlab router replay-prepare` compiles it into one replay-v5
JSONL file; see [docs/replay.md](docs/replay.md#where-the-replay-file-comes-from).
Session selection, think-time modes and file format are in the same document.
Other flags and run modes: [docs/detailed-modes-of-benchmark.md](docs/detailed-modes-of-benchmark.md).

### Downloading captures

Captures are published as scratch images holding just `/replay.jsonl`, tagged
`replay-<label>-<sha12>`, and pulled anonymously with `curl` (no Docker engine, no
registry client). Both are replay-v5 (per-request `think_ms` and a session index):

| Tag | Window | Sessions / requests | Download / on disk |
|---|---|---|---|
| `replay-40d-oct14-337a03810637` | 2026-09-05 → 10-14 | 3,190 / 370,658 | 1.3 GB / 8.8 GB |
| `replay-160d-oct15-dfe3662680de` | 2026-05-12 → 10-15 | 14,305 / 1,729,396 | 6.1 GB / 40.8 GB |

The 40d capture is the one embedded in the image and run by the Helm chart; it was
cut from the 160d one with `kvlab router replay-trim --last-days 40`. The 160d
capture is the full corpus, for runs that need more sessions or older traffic.

```
REPO=weka.io/kvlab; TAG=replay-40d-oct14-337a03810637   # or the 160d tag
TOKEN=$(curl -s "https://quay.io/v2/auth?service=quay.io&scope=repository:$REPO:pull" | jq -r .token)
LAYER=$(curl -s -H "Authorization: Bearer $TOKEN" \
  -H 'Accept: application/vnd.oci.image.manifest.v1+json' \
  "https://quay.io/v2/$REPO/manifests/$TAG" | jq -r '.layers[0].digest')
curl -L -H "Authorization: Bearer $TOKEN" "https://quay.io/v2/$REPO/blobs/$LAYER" | tar -xz replay.jsonl
```

## Verify mode and performance mode

The **default run has no verification** and gives the more precise performance
numbers: nothing is injected into the prompts and `ignore_eos` stays on, so output
volume is deterministic.

`--verify` (router replay only) checks cache coherency during the run. It injects
one deterministic UUID-style marker per user turn, asks later responses to recite
a bounded window of them, and scans every response for a marker it never sent. It
reports `PRESENCE_MISS` (the model did not repeat its own marker) and
`CROSS_CONTAMINATION` (a response carried another session's marker) beside the
throughput numbers.

- The run **stops on the first contamination event**, freezing the fleet in the
  state that produced it; `--verify-continue-on-contamination` keeps it going.
  Either way the event prints in full with the `--replay-series-indices` pair that
  reproduces it.
- `--verify` turns `ignore_eos` **off**, because forcing generation past the stop
  token pads responses with degenerate text a coherency check should not score.
  Output conformity then sits near 90%. `--verify-force-eos` keeps `ignore_eos`
  on when deterministic output volume matters more.
- Corrupt responses are captured (`--dump-garbage`, on by default); `--dump-dir`
  and `--dump-miss-dir` capture more.

Helm: `verify.enabled`, `verify.continueOnContamination`, `verify.forceEos`.

In most cases a verify run is representative for performance comparison. It can
miss effects at cross-request boundaries, where a request continues from the tail
of the previous one; a run without `--verify` aligns better there.

## Router

`kvlab router serve` is a model-aware LLM router: it matches each request's model
to a pool of endpoints and routes **by cost**, sending each request to the usable
backend with the lowest cost (the tokens it must prefill there, since that backend
does not hold the prompt's prefix, plus the context tokens it is already serving),
so a conversation keeps landing on the backend that holds its KV cache. Capacity
limits decide which backends are usable.

```
kvlab router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --backends 'http://vllm-a:8000|http://vllm-b:8000|http://vllm-c:8000'
```

Shapes supported (CLI and Helm for each in **[docs/router.md](docs/router.md)**):

- **One fleet**: `--backends` or pod selectors (`pods:app=vllm`).
- **Per-model routes**: `--route 'big,70b => http://a:8000|http://b:8000'`; both
  OpenAI-format and Anthropic-format requests follow the same rules.
- **Self-hosted models with a hosted fallback**: named models to your fleet,
  everything else to `--default`.
- **A single local endpoint**, with the model name resolved for you, for example
  to put Claude Code in front of a local server.

Capture, then replay: `--capture redacted` records traffic under
`~/.kvlab/router/capture/redacted/`; `kvlab router replay-prepare` turns it into a
replay file for the benchmark above. Authentication, per-user prefixes, metrics and
probes are all in [docs/router.md](docs/router.md).

## Results and reports

`--save-request-data ./results` writes per-request JSONL under
`./results/<run-timestamp>/` and an interactive `report.html` (TTFT, response time
and cache hits over time) at the end of the run. When the engine reports them, each request records where its prompt tokens came from (GPU,
host, external cache, computed) and engine-side timings, and the report's
Timeline view draws one row per session with a bar per request. `--live-port 8088`
serves the timeline live during the run. Both vLLM and SGLang are supported for
this and for cache-source metrics sampling. Details:
[docs/replay.md](docs/replay.md#results-kv-details-and-the-timeline).

### Regenerating and comparing reports

`visualize` rebuilds the report of one run. `visualize-merge` puts several runs
into one report, as arms of a head-to-head comparison:

```
kvlab benchmark visualize ./results/<run-timestamp> --timeline

kvlab benchmark visualize-merge ./results/run-a ./results/run-b \
  --labels baseline,candidate --baseline baseline --timeline -o ./results/compare
```

Each writes a new `report_vN.html` beside any existing report; `--overwrite`
replaces `report.html` instead. `--timeline` embeds the Timeline view.
`--max-elapsed 7h45m` drops records past that time from each run's own start, so
arms of different lengths compare over the same window.

For `visualize-merge`, `--labels` names the arms in argument order, and
`--baseline` picks the arm the summary's % comparison is relative to (default: the
slowest arm by completed requests). `--all <parent>` merges every subdirectory of
a parent directory. Without `-o`, the output directory is generated next to the
input.

## Helm quick start

`chart/kvlab/` is a run-once chart that runs the embedded 40d capture (3,190
sessions) with `kvlab benchmark auto --router-replay-file /kvlab/replay.jsonl`.
The chart defaults to `replay.realtime: true`, so the run is the real-time recipe
above. The only value most installs set is `endpoint`; get `<vX>` from the
[releases page](https://github.com/weka/kvlab/releases) (the chart `--version` is
mandatory).

```
helm upgrade --install my-replay oci://quay.io/weka.io/helm/kvlab --version <vX> \
  --set endpoint=http://10.71.0.4:8000
```

Smoke test, shortening `duration` (maps to `--timeout`, default `8h`):

```
helm upgrade --install my-replay oci://quay.io/weka.io/helm/kvlab --version <vX> \
  --set endpoint=http://10.71.0.4:8000 --set duration=3m
```

After the run the pod sleeps so results stay reachable with `kubectl exec` /
`kubectl cp`; delete the pod and the Deployment restarts the benchmark.

**Persisting results.** Results are always written to
`/results/kvlab-requests-data/<run-timestamp>/`. With nothing set that is an
`emptyDir` that goes away with the pod. To keep them, mount an existing PVC:

```
helm upgrade --install my-replay oci://quay.io/weka.io/helm/kvlab --version <vX> \
  --set endpoint=http://10.71.0.4:8000 \
  --set resultsClaim=my-existing-pvc
```

or `--set createResultsClaim=true` to have the chart provision `<release>-results`
(kept on `helm uninstall`). `resultsSubPath` changes the subdirectory. Every value
is documented inline in [`chart/kvlab/values.yaml`](chart/kvlab/values.yaml).

**Bare-URL model selector.** A bare `http(s)://` URL given as `--models` (or
`endpoint`) is all kvlab needs. If the URL has no path,
`GET <url>/v1/models` decides whether `/v1` is the base, and the first listed
model id is used unless `model=` is given (`model` in the chart).

## Releases and license

**Install commands for every version live on the
[releases page](https://github.com/weka/kvlab/releases)**: each release description
carries copy-paste `helm install`, `docker pull` and `go install` snippets pinned
to that version.

Embedding kvlab, the Dagger-based publishing flow and the repository layout are
in [docs/development.md](docs/development.md).

Apache-2.0, see [LICENSE](LICENSE).
