# The router

`kvlab router serve` is a model-aware LLM router. It matches each request's
model to a **pool** of interchangeable endpoints, then chooses among them by
prefix-cache affinity — so a conversation keeps landing on the backend that
already holds its KV cache instead of being spread by a load balancer.

It is one subcommand of the `kvlab` binary. There is no separate router program.

## Concepts, briefly

**A route maps model names to a pool.** Rules are first-match-wins, so specific
rules come first and a catch-all last:

```
--route '<patterns> => <endpoint>[|<endpoint>...] [as <model>]'
```

Patterns are comma-separated case-insensitive substrings of the model name, or
`*` for the catch-all. Endpoints are pipe-separated — the same syntax the kvlab
client already accepts for a multi-endpoint model, meaning the same thing on
both sides.

**A pool of one endpoint is a plain proxy. Several get affinity.** Same code
path either way; there is no mode to switch.

**Routing is by model, not by wire format.** Both OpenAI (`/v1/chat/completions`)
and Anthropic (`/v1/messages`) bodies carry a `model` field, and the same rules
apply to both. Both surfaces are the dialect's own, so both get **prefix-cache
affinity** — an Anthropic-format conversation concentrates on the backend already
holding its KV exactly as an OpenAI one does. Any path the dialect does *not*
claim is still forwarded to the matched pool unchanged, by load rather than by
prefix, which is what lets one router front a local fleet and a hosted API at
once.

**An upstream the router cannot probe is proxied transparently.** A route with a
single endpoint that answers neither `vllm:` metrics nor a model listing — a
hosted API — becomes a plain proxy: whatever the provider answers is relayed
verbatim, and the router applies **no circuit breaker, no retry, no routing
policy and no model rewriting** to it. Only a fatal failure to reach the upstream
at all — refused connection, DNS failure, timeout before the first byte —
produces a router-authored answer, and that is a `502`.

The reasoning is that each of those mechanisms is a judgement about a fleet the
router manages, and none can be made about somebody else's service. A breaker
answers 503 in place of the provider's own status and its `retry-after`; a retry
bills a metered API twice for one request; affinity has no choice to make with one
upstream. It is derived, never configured, and the startup log names every route
it applied to.

**Two unmanaged endpoints are still managed.** The condition is one endpoint, not
"is hosted": with two, the router picks between them, and it then needs the
breaker to notice one has died and the retry to reach the other. Passive health
never changes on its own, so dropping both there would leave a dead endpoint in
rotation permanently.

**Endpoint kind is discovered, once.** An endpoint serving `vllm:` metrics at
`/metrics` is treated as a vLLM instance: health-probed actively and eligible for
upstream metric aggregation. Anything else falls back to passive health — still
served, health inferred from real traffic rather than from probes it would always
fail. The probe never repeats. Because it keys on metric names rather than wire
format, **a vLLM fronted with an Anthropic API is still recognised as vLLM**.

**An endpoint is a URL or a pod selector.** One grammar, used everywhere an
endpoint can appear, and the two forms mix in one pool:

```
http://vllm-a:8000              a URL
pods:app=vllm                   every matching pod IN THE ROUTER'S NAMESPACE
pods:app=vllm:http              ... using each pod's containerPort named "http"
pods:app=vllm:8000              ... using 8000 for a pod declaring no port
http://legacy:8000|pods:app=vllm    both, which is what a migration looks like
```

Each discovered pod contributes **its own declared port**. That is the reason to
prefer a selector over a Service, and it is not a detail: a fleet run as several
DaemonSets — one per GPU topology, or per model — commonly listens on a different
port per set. A Service maps one port, so covering three sets needs three
Services, and the single selector that made them one pool is gone.

The port belongs to the route because it describes the *pool*. A router fronting
two fleets on different ports could not say so while this was a router-wide flag.
A colon is safe as the separator: Kubernetes label keys and values never contain
one.

Discovery searches the router's **own namespace** only. There is no syntax for
another, deliberately — reading pods across namespaces needs a ClusterRole, and a
standing cluster-wide pod read on every router is a poor trade for a case a
second router covers. It is what keeps the chart's RBAC a plain Role, which it
creates by default.

Discovery only ever *proposes* backends: the registry decides admission and
health decides eligibility, so a discovered pod is not routed to until it passes
the same checks a statically named one does, and a pod going NotReady leaves the
pool on its own.

**Routing is by cost.** By default each request goes to the usable backend with
the lowest cost: the tokens it would have to prefill there (the part of the
prompt that backend does not already hold) plus the context tokens it is
already serving. Ordinary requests have no tunable; the legs of disaggregated
serving (P/D legs and prefill-split jobs) add `--move-cost-weight`, below.
"Usable" is decided by the capacity
signals below, so these still apply under cost routing:
`--max-node-concurrency` (and its wait queue and
`--max-node-concurrency-queue-depth`), `--hard-ttft-split`,
`--rebalance-ratio`, `--retry-time-limit`, `--cache-refusal-ttl` and
`--cache-tail-ttl`. `--no-cost-routing` selects the older affinity ladder
instead, and is **required** by the settings only the ladder honours:
`--soft-node-concurrency`, `--soft-ttft-split`, `--relative-ttft-split`,
`--relative-decode-split`, `--transient-fallback-threshold`, a non-default
`--cache-split-guard` and the `--pd-prefill-soft-node-concurrency`
/ `--pd-decode-soft-node-concurrency` bands. `--pd-prefill-backends` works under
either mode. Setting one without `--no-cost-routing` stops the router at
startup, naming the flag. `--cost-routing` is accepted and does nothing.
Everything below about splitting, the split guard and the soft band describes
the ladder.

**Capacity is signalled, not assumed.** The backend's own `429` is the ultimate
signal and is always on. `--max-node-concurrency` is an opt-in early warning
that saves a round trip: set it to the backends' vLLM `--max-num-seqs` and
saturation is predicted rather than discovered one refusal at a time.

When a request's one preferred backend is saturated *only* by
`--max-node-concurrency` — not by a real `429` and not by `--rebalance-ratio`,
neither of which a slot freeing up on that backend fixes — and
`--retry-time-limit` is set, the request does not poll the whole fleet. It
joins a small FIFO queue bound to that one backend (depth equal to the
concurrency limit by default, one waiter per slot — see
`--max-node-concurrency-queue-depth` below) and is woken the instant a lease
there releases, then re-decides fresh. This is what keeps a queued request on
the backend that already holds its prefix instead of either spilling it to a
colder one or re-polling blind every few milliseconds. A release that frees a
slot nobody queued for, or a woken request that ends up routed elsewhere, both
correctly reach the next waiter in line — nothing is stranded.

By default the two flags above fully configure it; there is no need for a
separate queue depth or timeout. A queue already at depth answers `503`
immediately, since there is no instant that waiting would resolve. A queued
request that outlasts `--retry-time-limit` answers the same `429` it always
would have. Watch `router_backend_queued` (current depth),
`router_backend_queue_wait_seconds` (how long waiting actually took, by
outcome) and `router_backend_queue_full_total` (how often the depth bound
itself was hit).

`--max-node-concurrency-queue-depth` overrides that depth bound independently
of `--max-node-concurrency`'s own value. `0` (the default) is the behaviour
above, unchanged: depth equals the concurrency limit. A positive value sets
the depth explicitly — useful when a real test wants requests that arrive at
full backend capacity to actually wait rather than risk an early `503` from a
depth bound that is too tight; set it to a large number to make the queue
effectively unbounded for that run. It is still a bound, not true unlimited
queueing — the limit exists to protect router memory, not as an arbitrary
default — so treat a very large value as a deliberate, temporary choice for a
test rather than a production setting. It only does anything when
`--max-node-concurrency` is also set: at `0` the signal that produces a
queueable refusal is never raised, so this flag is inert on its own.

`--soft-node-concurrency` (ladder only; needs `--no-cost-routing`) makes that limit the **top of a band** rather than a
single cliff. Below the soft value a holder is an ordinary cache hit; at or
above it the router prefers to split the request elsewhere; and if nothing
clears `--cache-split-guard`, the request goes **back to the least-loaded
holder** and queues there until the hard limit.

That last step is the point, and it is the opposite trade from
`--transient-fallback-threshold`. Both relieve the same moment — the guard
refusing a duplicate — and they pay opposite prices:

| | keeps | pays |
|---|---|---|
| `--soft-node-concurrency` (stretch) | the cache hit; the KV is already there | queueing on a busier backend |
| `--transient-fallback-threshold` (overflow) | a short queue | a full prefill on a backend holding none of the prefix |

When both are set the stretch wins, because keeping the KV is the reason to
have the soft limit at all. They also confound each other in measurement, so
evaluate one at a time.

Pick the pair from where the fleet actually sits: a soft limit below the load
backends idle at fires continuously and a hard limit above their ceiling never
binds. `router_cache_stretch_inflight` is the check — piled against the hard
limit means soft is set too low, and the router is paying the queueing cost
continuously rather than as relief.

**Heavy prefills move to their own nodes.** With `--prefill-split` (works under cost routing and the ladder; the prefill node and the decode resend are chosen by the active mode), a backend
that refuses a request because recomputing its prompt would stall the node's
running decodes (HTTP `503` carrying `X-Weka-Prefill-Reject: <tokens>`, error type
`large_prefill_rejected`; it is a routing signal and never counts toward the
circuit breaker, backend health or 5xx accounting; the engine's `WEKA_PREFILL_REJECT_TOKENS`) does not fail the
request. The router sets a node aside, runs the prefill there as a copy of the
request limited to one token (`max_tokens=1`, `stream=false`), then sends the
original through ordinary routing over the remaining nodes. The client never
sees the refusal. Both carry `vllm_xargs.weka_prefill=1`, which the engine
never refuses, so a rejection cannot loop; if the shared KV tier lagged the
prefill, the decode node recomputes the rest. With the flag off the `503`
reaches the client unchanged.

Under cost routing the prefill nodes are routed separately from the decode nodes,
as with fixed P/D. Prefill jobs are placed by a flow of their own with its own
prefix tree: among the prefill nodes with budget room (and under the node
concurrency bound) the one with the lowest cost wins, not the fullest node (the
ladder still packs onto the fullest and uses the flow only to break ties). Cost
is weighted as for the fixed P/D prefill leg: `--move-cost-weight` times the
prompt tokens that node cannot supply, plus the node's pending job tokens
(the jobs queued on it and running there, as charged by the manager). The
node's active context is left out, for the same reason as on a P node. What a
node can supply is the larger of what its prefill tree holds and what an
earlier job stored in the tier: the deepest prefix of the prompt any node's
prefill tree holds, cut to `floor(depth / unit) * unit` by
`--prefill-store-unit-tokens`, is credited to every prefill node without being
recorded. So with the tier holding a conversation's earlier jobs a job goes to
the node with the shortest queue, and a node holding more than the cut keeps an
advantage only for the tokens past it; with nothing stored a conversation stays
on its prefill node until that node's queue is longer than another's by more
than `--move-cost-weight` times the prefix it holds. Prefixes the tier holds from decode nodes' own
stores are not credited: the router has no record of them. On equal cost a job
(and a fixed P/D prefill leg) goes to the node holding more of its prompt
itself, since a local hit is cheaper than a tier load. `cost_move` in
`router_route_decisions_total{pool="<pool>:prefill"}` counts jobs placed away
from the node holding the deepest local prefix, credited or not. After a job succeeds its prefix is
recorded in the prefill tree only, never in the decode tree, so the decode resend
(over the decode nodes, with the decode tree) is unaffected. A node's marks count
only while it is in the role that made them: when a node leaves decode its decode
marks are dropped, and when it returns to decode its prefill marks are dropped.
`--prefill-release-idle` therefore also sets how long a node keeps its prefill
prefixes. Conversion and release stay the manager's decisions (budget, fleet fraction,
`--prefill-convert-after`, `--prefill-release-idle`); cost only chooses among the
current prefill nodes and never converts one. The flow also hears of a prefill
node's `429` refusals and later accepts, like the fixed P/D flow. The ladder keeps
choosing jobs with the shared flow.

After a successful job the decode resend is credited for what the job stored: the
leading `floor(prompt / unit) * unit` tokens are treated as held by every decode
node in that request's cost, and are not recorded in the decode tree. A
non-holder pays only for tokens past the cut; a node whose decode tree holds more
keeps an advantage only for what it holds beyond the cut. `--prefill-store-unit-tokens`
(default 0 = the whole prompt) sets `unit`, the granularity of the shared KV tier.

A node is `decode` (normal), `draining` (no new decode routing; its running
requests finish, and prefill jobs queue behind them), `prefill` (jobs only), and
back to `decode` once it has no prefill job running or queued — at once, or
after `--prefill-release-idle` without one, so a burst that follows soon after
finds it ready instead of draining a node afresh. A draining node whose jobs
all went away returns at once regardless: it serves nothing while it waits.
A job goes to the prefill node with the most queued plus in-flight prefill that
still has room under `--prefill-node-budget-tokens`; the ordinary routing flow
only breaks a tie. Packing onto the fullest node lets the others run dry and
return to decode, where spreading jobs would keep every prefill node a little
busy and none would ever go back. When none has room a decode node is drained, up
to `floor(nodes * --prefill-max-fleet-fraction)` and always leaving one decode
node. With `--prefill-convert-after` set, that drain waits until the
least-loaded prefill node has been over budget for that long, jobs queueing on it
meanwhile: draining costs decode capacity before it adds any prefill, so a
brief overflow is cheaper to queue through. Successive drains are also at least
`--prefill-drain-min-interval` apart (default 5 s, both split modes; 0 = no
limit). An overflow from an over-estimated charge, or from a burst that finishes
quickly, gets that long to clear before another decode node is given up. Jobs
queue on the prefill nodes meanwhile, and `router_prefill_drains_deferred_total`
counts the jobs held back. The pool's first prefill node is never delayed.

In both modes a draining node takes no job while a prefill node exists, since
a job there would wait for the node's decodes to finish. The job queues on the
least-pending prefill node instead, and moves to the drained node once that
node is a prefill node. A drained node with nothing in flight takes the job at
once. A drain is abandoned only once the overflow that called for it has
cleared; until then the node keeps draining with no job on it. Every prefill
job's charge and router wait are in `router_prefill_job_tokens` and
`router_prefill_job_wait_seconds`. Each step of the control loop logs a line:
`decode node draining for prefill`, `draining node is now a prefill node` (with
`drain_took`), and `prefill node released to decode` (reason `idle`,
`low_delay` or `drain_abandoned`). Past the cap, or while the overflow
is younger than that, jobs queue on the least-loaded prefill node, and with no
prefill node at all the request is resent to the backend that refused it, with
the bypass. The budget also bounds what a node is *sent*: a node runs jobs only
while their tokens fit its budget (a single larger job runs alone), so jobs
queued over it wait in the router, not in the engine. A waiting job moves to
any prefill node that could start it at once and has no queue of its own — one
that frees room, or finishes draining — taking the head of the longest queue
first (`router_prefill_job_moves_total`). A burst that arrives while
`--prefill-convert-after` holds back conversions therefore spreads over the
nodes converted after it, instead of running on the first one for minutes while
they sit idle. It needs a KV tier the nodes share; without one the prefill is
wasted work.

**Prefill/decode disaggregation (vanilla vLLM NIXL).** `--pd-prefill-backends=<urls>` (works under cost routing, the default, and the ladder)
names the prefill (P) nodes by URL; every other backend is a decode (D) node.
With a static `--backends` list the URLs must be members of it, or the router
exits at startup. With `--backends=pods:<selector>` the members are not known at
startup, so there is no such check: roles are assigned by URL as pods appear
(discovery builds `http://<PodIP>:<port>`, so name the P pods that way), and a
URL that has not appeared 30 s after start, or later leaves the pool, is logged
as a warning (and logged again when it returns), never an exit. Until a P node is
live, requests fall back to a D node alone; with no live D node a request is
answered `503` at once ("no healthy decode backend is available (only prefill
nodes are live)" when only P nodes are up). Selecting the P pool by its own label
selector is not supported. It replaces vLLM's round-robin toy proxy. With `--prefill-split` it needs
`--prefill-split-mode=pd`, where it names the dynamic P set's initial nodes (below). Every `/v1/chat/completions` and `/v1/completions` request goes
to a P node first, as a copy with `max_tokens=1`, `stream=false`, no
`stream_options` and `kv_transfer_params={"do_remote_decode": true, ...}`. The
`kv_transfer_params` object in the P node's answer goes, verbatim, into the
original request (streaming preserved), which then goes to a D node. Both legs
carry the same `X-Request-Id` and the route's model rewrite.

Each leg is chosen by the ordinary routing flow (prefix affinity plus load, with
`--soft/--max-node-concurrency`), restricted to its own node set, and each set has
a prefix tree of its own: a conversation's turns return to the P node whose prefix
cache has them, and to the D node that already holds the blocks, so NIXL moves
only the missing ones. The prefix is recorded for the node that served each leg.
P legs count as in-flight on their node; a saturated P set or a `429` is waited
out and retried (bounded by `--retry-time-limit` when set). A prefill leg that
fails, or answers without `kv_transfer_params`, sends the request to a D node
alone, which recomputes (`router_pd_fallbacks_total`). `/v1/messages` carries no
`kv_transfer_params` and is served on D nodes only. `/router-viz` shows each
backend's role as its `mode` (`prefill` or `decode`).

Under cost routing each leg picks the usable node of its own set with the lowest
cost, over that set's own prefix tree, with `w` = `--move-cost-weight` (default
4):

- P leg: `w × (prompt tokens not held in the P tree) + the node's in-flight
  (queued and running) uncached prefill tokens`. The node's active context
  tokens are left out: on a P node they are the whole prompt of every queued
  leg, cached part included, so one extra queued leg would outweigh a whole held
  prefix.
- D leg (one carrying `kv_transfer_params`): `w × (tokens the D tree lacks) +
  in-flight prefill + active context tokens`. The lacking tokens arrive by NIXL
  transfer rather than recompute.

So a conversation leaves the node holding its prefix only when that node's
extra queue (P) or load (D) exceeds `w` times what moving costs: the recompute
it would put on another P node's queue, or the KV another D node would pull and
hold again. `w = 1` is a greedy time-to-first-token choice, and on a
compute-bound P set without a shared KV tier it is unstable: each move puts a
whole-prompt recompute on another node's queue, which lengthens the queues that
make the next move look worthwhile, and the fleet settles where most turns move
and most prompts are recomputed. Above 1 a move must also pay for the delay its
extra work imposes on the legs queued behind it; at 4 a replay of live-sized
traffic (50k-250k-token prompts, 1-3k-token turns, P nodes draining a fixed
token rate) keeps 98-99% of turns on their P node where `w = 1` keeps 63%
(`TestPrefillLegStaysOnItsHolderOnACompoundingFleet`). New conversations hold
nothing anywhere and still go to the shortest queue. On D, without the weight,
one conversation's active context (a whole prompt) outweighs the prefix a D node
holds, so turns churn between D nodes and each move pulls the whole prompt's KV
again; with it, a D node keeps its conversations until it carries several
conversations' worth more than another. A node that is slow rather than loaded
does not show in these token counts; `--hard-ttft-split` covers that.
`router_route_decisions_total{pool="<pool>:prefill",decision=~"cost_stay|cost_move"}`
(and `pool="<pool>"` for D) shows stickiness: `cost_move` is a leg sent away from
the node holding the deepest local part of its prefix.
Per-role `--pd-*-max-node-concurrency` caps are hard limits in both modes. The
per-role *soft* bands are ladder-only.

The two roles can run different concurrency bands:
`--pd-prefill-soft-node-concurrency` / `--pd-prefill-max-node-concurrency` for the
P flow and `--pd-decode-soft-node-concurrency` / `--pd-decode-max-node-concurrency`
for the D flow, which also governs requests that bypass P/D (`/v1/messages`). Each
defaults to 0, inheriting the global `--soft-node-concurrency` /
`--max-node-concurrency`, and is an error without `--pd-prefill-backends`. A
decode-only node carries far more concurrent requests than a mixed one, so a shared
band tends to cap the decode nodes first. The startup log prints
`signals="prefill=[...] decode=[...]"` with each role's band. The `router_cache_*` series
for the P flow carry `pool="<pool>:prefill"`.

**Dynamic vanilla P/D.** `--prefill-split --prefill-split-mode=pd` runs the
prefill split above as NIXL prefill/decode disaggregation. The default,
`--prefill-split-mode=reject`, is the split as described above.

- **Every** `/v1/chat/completions` and `/v1/completions` request is first a
  prefill leg, with the same body as fixed P/D: `max_tokens=1`,
  `kv_transfer_params.do_remote_decode`. It then goes to a decode node with the
  returned `kv_transfer_params`, which pulls the KV over NIXL. No `503`, no store,
  no resend.
- **What drains a decode node: `--prefill-drain-delay` (default 2 s).** A
  prefill node mostly loads KV from the shared tier, and a compute-token budget
  does not see that load. The time a leg waits for a prefill node does.
  - The signal is the 90th percentile of the legs' router wait: from placement
    to the slot that lets a leg be sent. It covers legs granted in the last 10 s,
    plus legs still waiting counted at their age so far.
  - Drain: legs are waiting now and the signal has stayed above the delay for
    `--prefill-convert-after`. One decode node is drained, within the cap and
    `--prefill-drain-min-interval`, and the clock restarts. The decision reads
    only the load on the current prefill nodes; drains already under way are
    not counted as capacity on its way.
  - The node drained is the decode node whose decodes should end soonest: one
    with nothing in flight first, then the one whose oldest in-flight request
    started last, then fewer requests in flight. A drain takes no prefill leg
    until its decodes finish.
  - Abandon: a drain no longer needed returns to decode at once. While nothing
    waits, or the signal is at or below half the delay, the newest draining
    node with no job on it goes back, one per tick (100 ms). The threshold is
    below the drain condition, so a signal hovering at the delay does not drain
    and abandon in turn. A draining node keeps its decode prefix marks until it
    becomes a prefill node, so an abandoned drain returns with its
    conversations' decode affinity intact; it only paused new decode legs.
  - Release: nothing waiting, or the signal at or below a quarter of the delay,
    for `--prefill-release-idle`. The newest converted prefill node with nothing
    queued goes back to decode, and that clock restarts too. Never below
    `--prefill-min-nodes`.
  - **Placement packs.** A leg goes to the first prefill node with room in a
    fixed oldest-first order: the initial nodes, then the others by when they
    were converted, then by URL. The oldest nodes fill first, the newest runs
    dry, and it is the first the release rules return. Decode legs stay
    cost-routed.
  - `--prefill-node-budget-tokens` and the prefill band's
    `--pd-prefill-max-node-concurrency` only bound what is sent to one node. Set
    at least one, or legs never wait in the router and nothing drains. Prefill
    nodes still queueing in the engine then show only in
    `router_pd_prefill_service_seconds`.
  - `--prefill-drain-delay 0` drains on the budget instead, as the reject mode
    does.
- **Charge diagnostics.** For one conversation in 64, every prefill leg logs a
  `pd leg charge` line with these fields:
  - `conversation`, `units`, `prev_units`;
  - `diverge_unit`: the first unit of the prompt that differs from the
    conversation's previous leg, as the router's tree sees it, or -1 on the
    first leg;
  - `shared_with_prev_tokens`;
  - `total_tokens`, `prefill_tree_tokens`, `decode_tree_tokens`,
    `stored_tokens`, `charge_tokens`.

  A conversation is named by the request id's part before `-t<turn>-` (kvlab's
  replay ids). Requests whose ids have another shape are not sampled. Comparing
  `charge_tokens` with the engine's `local_compute` shows whether the charge
  tracks real compute, and `diverge_unit` shows where it stops.
- Each prefill leg is a prefill job of the split's manager. It is charged what a
  prefill node must compute: the prompt less the larger of two credits.
  - The deepest prefix a prefill node holds locally.
  - What the shared KV tier holds: the deepest prefix either the prefill or the
    decode tree has recorded, since every earlier leg stored its prompt there,
    cut to `--prefill-store-unit-tokens`.

  The trees forget a prefix after `--cache-tail-ttl`, so with turns further
  apart than that a leg is charged its whole prompt; raise it, which costs only
  router memory.
  Conversion, drain, release and placement are the split's, unchanged:
  - Budget: `--prefill-node-budget-tokens`.
  - Drain: `--prefill-convert-after`.
  - Release: `--prefill-release-idle`.
  - Cap: `--prefill-max-fleet-fraction`.
- A drained node takes legs only once the decodes it had have finished. So no
  node runs prefill legs next to decodes it took as a decode node; the engine's
  `VLLM_MAX_PREFILLS_EXEMPT_PREFILL_ONLY` assumes exactly that.
- `--pd-prefill-backends`, optional here, names the nodes that start as prefill
  nodes. They are released like any other. `--prefill-min-nodes` (default 1) is
  the fewest prefill nodes kept: none is released below it, and decode nodes are
  drained to reach it, at start when no node is named. Every prefill node counts
  toward `--prefill-max-fleet-fraction`.
- Prefill legs run under the prefill band (`--pd-prefill-max-node-concurrency`,
  else `--max-node-concurrency`).
- Decode legs are placed by cost routing over the decode nodes. A draining or
  prefill node never takes one. With no decode node left the request is answered
  `503`.
- A failed leg, or one answered without params, sends the request to a decode node
  alone, which recomputes (`router_pd_fallbacks_total`,
  `router_prefill_jobs_total{outcome="failed"}`).
- `/v1/messages` is served on decode nodes only.
- Engines: every node is launched the same way, `kv_role=kv_both` with the P/D
  connector in token boundary mode, so any node serves either leg.
- Metrics are the split's: `router_prefill_nodes{mode}`, conversions, releases,
  jobs (one per request), per-backend mode and pending tokens. The fixed-P/D leg
  metrics come on top.
- Each drain and release logs one line: `decode node draining for prefill` /
  `prefill node released to decode`. It carries the reason, the node's in-flight
  and pending tokens, the prefill nodes' pending, running and queued tokens and
  queued jobs, nodes per mode, the budget, and the leg's tokens.
- `/router-viz` shows `decode`, `draining` and `prefill` per backend.

As in the reject mode, cost routing spreads legs over the prefill nodes. Under
steady load a node with a little work never goes idle, so it is not released
until traffic drops.

`--rebalance-ratio` is **off by default**. Set to `0.5` it makes a backend
carrying more than twice the fleet minimum stop taking new work, so the idle
capacity beside it gets used — but that trades locality for evenness, and a
fleet where affinity is working is *supposed* to look imbalanced. Concentration
is the mechanism, not a fault in it. Repeated measurement found spreading it
away buys no throughput while costing cache hits, so it stays off unless a
deployment values evenness for its own sake.

If you do turn it on: a backend under **8 in-flight** is never rebalanced away from, whatever the
proportions say. The floor is fixed, and it is what keeps the ratio meaningful:
the fleet minimum is 0 whenever any backend is momentarily idle, and then
`(inflight − 0)/inflight` is 1.0 for every backend carrying anything at all. A
fleet of `1,1,1,0` would read as imbalanced as `20,20,20,0` — they are
ratio-identical, and only magnitude separates them. Above the floor the ratio
decides; below it nothing is under pressure and moving a prefix costs more than
the imbalance does.

---

## Use case 1 — one vLLM fleet, all traffic

The simplest deployment. Every model goes to one pool, and affinity concentrates
each conversation on the backend holding its prefix.

### CLI

```bash
kvlab router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --backends 'http://vllm-a:8000|http://vllm-b:8000|http://vllm-c:8000'
```

`--backends` is shorthand for `--route '* => a|b|c'` — no model patterns, no
rules, every request to one pool. It takes pod selectors too, which is the
tidiest form there is:

```bash
kvlab router serve --listen :8080 --backends pods:app=vllm:http
```

Set `--max-node-concurrency` to the backends' vLLM `--max-num-seqs` to predict
saturation rather than discovering it one refusal at a time:

```bash
kvlab router serve \
  --backends 'http://vllm-a:8000|http://vllm-b:8000' \
  --max-node-concurrency 32
```

### Helm

```yaml
# values.yaml
router:
  backends:
    - http://vllm-a:8000
    - http://vllm-b:8000
    - http://vllm-c:8000
  signals:
    maxNodeConcurrency: 32
```

```bash
helm upgrade --install my-router \
  oci://quay.io/weka.io/helm/kvlab-router --version v1.2.3 \
  -f values.yaml
```

Backend URLs contain no commas, so `--set` is safe here and no values file is
needed:

```bash
helm upgrade --install my-router \
  oci://quay.io/weka.io/helm/kvlab-router --version v1.2.3 \
  --set router.backends[0]=http://vllm-a:8000 \
  --set router.backends[1]=http://vllm-b:8000 \
  --set router.signals.maxNodeConcurrency=32
```

The same pool from pod labels — still no patterns, still just `backends`. An
endpoint is a URL string or a selector map, and that is the same grammar a
route's `endpoints` takes:

```yaml
router:
  backends:
    - pods: {app: vllm}
      port: http
```

```bash
--set-string 'router.backends[0].pods.app=vllm' \
--set-string 'router.backends[0].port=http'
```

Use the map with `--set`. A label selector is comma-separated and `--set` splits
on commas, so the string `pods:app=vllm,tier=prod` arrives truncated at its first
label — silently, rendering fine and failing at runtime. In a values *file* the
string form is fine; the problem is `--set` specifically.

All forms render to the one CLI grammar, so the pod ends up running
`--backends pods:app=vllm:http` either way. A URL and a selector can share the
list, which is what a migration looks like:

```yaml
router:
  backends:
    - http://legacy-vllm:8000
    - pods: {app: vllm}
      port: http
```

Covered by `TestUseCase1_SingleFleetAllTraffic`.

---

## Use case 2 — per-model routes, both APIs

Different models on different fleets. Routing is by model name, so clients using
the OpenAI API and clients using the Anthropic API hit the same rules.

### CLI

```bash
kvlab router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --route 'fast,small => http://vllm-small-a:8000|http://vllm-small-b:8000' \
  --route 'big,70b    => http://vllm-large-a:8000|http://vllm-large-b:8000'
```

Both of these route to the `big` pool, because both name the model the same way:

```bash
curl :8080/v1/chat/completions -d '{"model":"big-70b","messages":[...]}'   # OpenAI
curl :8080/v1/messages         -d '{"model":"big-70b","messages":[...]}'   # Anthropic
```

Use `as <model>` when the client's name for a model differs from the backend's:

```bash
--route 'sonnet => http://vllm-a:8000 as Qwen/Qwen3-32B'
```

### Helm

```yaml
router:
  routes:
    - "fast,small => http://vllm-small-a:8000|http://vllm-small-b:8000"
    - "big,70b    => http://vllm-large-a:8000|http://vllm-large-b:8000"
    # `as <model>` rewrites the model before forwarding, same as on the CLI —
    # a route is the same string either way.
    - "sonnet     => http://vllm-a:8000 as Qwen/Qwen3-32B"
```

**With `--set`, use the structured form instead.** Helm's `--set` splits values
on commas, so a multi-pattern route written as one string is truncated at its
first pattern — it renders without error and fails at runtime. Given as lists,
no single value contains a comma:

```yaml
router:
  routes:
    - patterns: [fast, small]
      endpoints: [http://vllm-small-a:8000, http://vllm-small-b:8000]
    - patterns: [sonnet]
      endpoints: [http://vllm-a:8000]
      as: Qwen/Qwen3-32B
```

```bash
--set router.routes[0].patterns[0]=fast \
--set router.routes[0].endpoints[0]=http://vllm-small-a:8000
```

Both forms are accepted; a values file can use whichever reads better.

Covered by `TestUseCase2_PerModelRoutesAcrossBothAPIs`.

---

## Use case 3 — self-hosted models, hosted fallback

Named models go to your fleet; everything else falls through to a hosted API, so
a client can point at one base URL for both.

### CLI

```bash
kvlab router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --route 'llama,mistral => http://vllm-a:8000|http://vllm-b:8000' \
  --default 'https://api.anthropic.com using client' \
  --strip-auth-when 'llama,mistral'
```

Order matters: the specific rule is matched first, and `--default` is the
catch-all. `--strip-auth-when` drops the caller's credentials before forwarding
to an unauthenticated local upstream, so a client's key is not leaked into your
own logs.

The hosted endpoint has no `/health` or `/metrics`; the router works that out and
treats it as passive, so it is usable immediately. Pass `--passive-health` to
skip the probe entirely when you already know.

### The full shape

Everything at once: two local fleets discovered from pod labels on different
ports, a model-name rewrite, and Anthropic as the fallback paid for with the
caller's own key.

```bash
kvlab router serve \
  --listen :8080 --metrics-listen 0.0.0.0:29000 \
  --route 'fast,small => pods:app=vllm,size=7b:http' \
  --route 'sonnet     => pods:app=vllm,size=70b:http as Qwen/Qwen3-32B' \
  --default 'https://api.anthropic.com using client' \
  --strip-auth-when 'fast,small,sonnet' \
  --max-node-concurrency 48
```

What each line does:

- **`fast,small`** — either substring in the model name picks the 7b pool,
  discovered from pod labels, each pod on its own port named `http`.
- **`sonnet`** — a client asking for `sonnet` reaches the 70b fleet, with the
  model rewritten to the checkpoint the backend actually serves. The client
  never learns the local name.
- **`--default`** — anything unmatched goes to Anthropic. The caller's key is
  forwarded, so they pay for their own hosted calls.
- **`--strip-auth-when`** — the caller's key is removed before it reaches a
  local fleet, which is unauthenticated and would otherwise log someone's
  credential.
- **`--max-node-concurrency`** — the backends' vLLM `--max-num-seqs`, so
  saturation is predicted rather than discovered one refusal at a time.

Both APIs route through the same rules, so a client on `/v1/messages` and one on
`/v1/chat/completions` are matched identically — and both are cache routed. Watch
`router_route_decisions_total{decision="cache"}` against `decision="load"` to see
it: a fleet serving Claude-shaped clients should sit almost entirely on `cache`.

### Helm

```yaml
router:
  routes:
    - "llama,mistral => http://vllm-a:8000|http://vllm-b:8000"
  default: "https://api.anthropic.com using client"
  stripAuthWhen:
    - "llama,mistral"
```

The full shape above, structured — which is what `--set` requires, since both
model patterns and label selectors are comma-separated:

```yaml
router:
  routes:
    - patterns: [fast, small]
      endpoints:
        - pods: {app: vllm, size: 7b}
          port: http
    - patterns: [sonnet]
      endpoints:
        - pods: {app: vllm, size: 70b}
          port: http
      as: Qwen/Qwen3-32B
  default: "https://api.anthropic.com using client"
  stripAuthWhen: ["fast,small,sonnet"]
  signals:
    maxNodeConcurrency: 48
```

Covered by `TestUseCase3_SelfHostedModelsWithHostedFallback`.

---

## Use case 4 — fronting a single local endpoint

Front a local model server on `http://127.0.0.1:25201` with the router on
`http://127.0.0.1:25202`, sending every model to that one upstream:

```bash
kvlab router serve \
  --listen 127.0.0.1:25202 \
  --default 'http://127.0.0.1:25201' \
  --strip-auth-when '*'
```

- `--default` is the catch-all rule (`* => <upstream>`): every requested model
  name matches. The request path and query are appended to the upstream, so a
  `POST /v1/messages` is forwarded to `http://127.0.0.1:25201/v1/messages`.
- `--strip-auth-when '*'` drops the inbound `Authorization`/`x-api-key`
  headers, which unauthenticated local servers (vLLM, llama.cpp, ...) otherwise
  reject.
- Append `as <model>` to pin the model rewrite and skip discovery:
  `--default 'http://127.0.0.1:25201 as my-local-model'`.
- Use repeated `--route '<substr>,<substr> => <upstream>[ as <model>]'` rules
  (first match wins) for per-model fan-out instead of one catch-all.
- `--capture redacted` records traffic under `~/.kvlab/router/capture/redacted/`
  (override with `--capture-dir`) for later `kvlab router replay-prepare`.

### Auto-model

A client's model name often means nothing to the upstream: Claude Code sends
`claude-*`, which a single-model server does not recognise (vLLM answers
`POST /v1/messages` with a bare `404`). On startup the router asks each upstream
that has no explicit `as <model>` what it serves (`GET <upstream>/v1/models`); if
the answer is exactly one model, every matching request's `model` is rewritten
to it:

```
auto-model: rewriting every matching request's model to what the pool serves pool=default model=zai-org/GLM-5.2-FP8
  route: * => http://127.0.0.1:25201 as zai-org/GLM-5.2-FP8 (auto-discovered) (strip-auth)
```

A multi-model upstream is left alone, so this cannot silently collapse a real
fan-out onto one model. `--auto-model` controls it: `auto` (default), `force`
(always take the first listed model), `off` (never probe).

The question is asked of the *pool*, not of one endpoint, and only once a backend
has passed a health check — so a pool whose first endpoint is dead still resolves
from any sibling that answers, and a backend still loading weights is waited for.
The answer takes effect the moment it lands, including for pools whose members
arrive later from pod discovery.

### Pointing Claude Code at it

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:25202
export ANTHROPIC_AUTH_TOKEN=dummy   # any non-empty value; stripped by --strip-auth-when '*'
claude
```

With `--user-prefix` on the router, each client tags itself via the first path
segment (`ANTHROPIC_BASE_URL=http://127.0.0.1:25202/alice`), and requests,
captures and log lines carry that user. See *Per-user path prefixes* below.

Add `--metrics-listen` to get the probes (`/liveness`, `/readiness`), `/metrics`
and `/router-viz`; see *Probes are on the metrics listener* above.

This needs an upstream that serves `/v1/messages` itself. One that speaks only
Chat Completions — vLLM's Rust frontend, for one, answers `501` — needs the
translation mode below.

## Use case 5 — Anthropic clients on an OpenAI-only fleet (`--anthropic-api`)

`--anthropic-api` makes the router serve the Anthropic Messages API itself, by
translating each request to the backends' `/v1/chat/completions` and the response
back. Claude Code works through it end to end — multi-turn tool use, streaming,
thinking — against a model that is not Claude.

```bash
kvlab router serve \
  --listen 127.0.0.1:8080 \
  --route '* => http://vllm-host:8000 using /path/to/backend.key' \
  --anthropic-api \
  --anthropic-model-map '*=deepseek-ai/DeepSeek-V4.1-Flash'

ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_MODEL=claude-sonnet-4-5 ANTHROPIC_SMALL_FAST_MODEL=claude-haiku-4-5 claude
```

A translated request is handed to the ordinary routing handler as a
`/v1/chat/completions` request, so it is authorised, prefix-affinity routed,
retried, queued and measured exactly like native chat traffic. Off (the default),
`/v1/messages` is proxied untranslated as before.

| Flag | Meaning |
|---|---|
| `--anthropic-api` | Turn the mode on: `POST /v1/messages` and `/v1/messages/count_tokens` (and `/<user>/v1/messages...` with `--user-prefix`) are translated. |
| `--anthropic-model-map '<pattern>=<model>'` | Rewrite the client's model before routing. Pattern = comma-separated case-insensitive substrings or `*`; first match wins; repeatable. Claude Code sends a main and a small/fast model name — `'*=<backend model>'` maps both. Unmatched names go as sent (a route's `as <model>` or auto-model may still rewrite them; auto-model does not run on a single passive upstream, so map explicitly there). Responses echo the client's name. |
| `--anthropic-thinking request\|always\|never` | When backend reasoning (`reasoning_content` / `reasoning`) is returned as `thinking` blocks. `request` (default): only if the request enabled thinking, as Anthropic does. Otherwise it is dropped. |
| `--anthropic-thinking-kwarg <key>` | `chat_template_kwargs` key set to the request's thinking on/off (`enabled`/`adaptive` → true, `disabled` → false); repeatable; default `thinking` (DeepSeek). A request without `thinking` leaves the backend's default. Empty sends none. |
| `--anthropic-max-tokens-cap N` | Clamp `max_tokens`. Claude Code asks for 32k output tokens on every call; a backend whose context cannot fit prompt + that refuses the request. 0 = as sent. |
| `--anthropic-debug-dir DIR` | Append every exchange — Anthropic request, translated request, raw backend response, translated response — to `DIR/anthropic-exchanges.jsonl`. Bodies only, never headers; contains full prompts. The way to tell a translation bug from model behaviour. |

**Credentials.** The client's `x-api-key` / `Authorization` authenticates to the
router (`--api-key`) and goes no further; the backend gets its route's own
credential (`using <file>`), or none.

### What is translated

| Feature | Support |
|---|---|
| Streaming | Full event sequence: `message_start`, `ping`, `content_block_start`/`_delta`/`_stop` (`text_delta`, `thinking_delta` + `signature_delta`, `input_json_delta`), `message_delta` (stop_reason, usage), `message_stop`. Blocks are strictly sequential; interleaved parallel tool-call fragments are buffered and each call emitted whole, in order. Pings every 10s of silence (hidden reasoning). A backend stream that ends without a finish reason ends in an `error` event, never a `message_stop`. A client disconnect cancels the backend request. |
| Non-streaming | Full message. A non-streamed backend answer to a streamed request is replayed as events. |
| Tools | `tools` → functions (`input_schema` → `parameters`); `tool_choice` auto/any(→required)/tool/none; `disable_parallel_tool_use` → `parallel_tool_calls:false`; `tool_use` ↔ `tool_calls` (ids round-trip); `tool_result` (string or blocks, `is_error` → `Error: ` prefix) → `role=tool`. Parallel calls supported. Server tools (web search, code execution, ...) are dropped: an OpenAI backend cannot run them. |
| Thinking | See the flags. History `thinking` blocks are dropped (their signatures mean nothing to the backend; reasoning templates strip prior reasoning anyway). Emitted thinking blocks carry an empty signature. |
| Images | `base64` and `url` sources → `image_url` parts; images inside a `tool_result` follow in the next user turn (a tool message carries text only). Needs a multimodal backend. |
| Documents | Text sources become text; PDFs and file references are replaced by a note saying so. |
| System | String or blocks; `cache_control` ignored (the backend's prefix cache needs no hint). |
| Sampling | `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` → `stop`, `metadata.user_id` → `user`. |
| Stop reasons | `stop` → `end_turn`, or `stop_sequence` (+ the sequence) when vLLM's `stop_reason` names one of the request's; `length` → `max_tokens`; `tool_calls` → `tool_use`; `content_filter` → `refusal`. |
| Usage | `input_tokens` = prompt − cached, `cache_read_input_tokens` = cached, `output_tokens`; `cache_creation_input_tokens` is 0. `message_start` carries an estimate; `message_delta` the real numbers. |
| Errors | Anthropic envelope with the type for the status (`invalid_request_error`, `authentication_error`, `not_found_error`, `rate_limit_error`, `overloaded_error` for 503/529, `api_error`). A backend context-length refusal is reworded `prompt is too long: ...`, which Claude Code acts on by compacting. |
| `count_tokens` | The backend's `/tokenize` when the pool routes it; otherwise the router's byte estimate of the translated prompt. |
| Ignored, never refused | `anthropic-version`, `anthropic-beta` (not forwarded), `context_management`, `service_tier`, `container`, `mcp_servers`, `citations`, unknown block types. |

**Not translated:** Batches, Files and Models API endpoints; prompt-caching
semantics (`cache_control` TTLs); server tools; `redacted_thinking` content.

---

## What is published

Every release publishes two images and two charts from the same commit under the
same semver, so a benchmark and a router from one release are known to agree.

| | image | chart |
|---|---|---|
| router | `quay.io/weka.io/kvlab-router:<version>` | `oci://quay.io/weka.io/helm/kvlab-router:<version>` |
| benchmark | `quay.io/weka.io/kvlab:<version>` | `oci://quay.io/weka.io/helm/kvlab:<version>` |

The two images are the same `kvlab` binary. The benchmark one additionally
embeds a multi-GB replay artifact so a benchmark pod can replay captured traffic
with no volume; the router never reads it, so the router image does not carry it.

The chart pins its image purely by propagation — `Chart.yaml` `appVersion` feeds
the deployment's `imageTag | default .Chart.AppVersion` — so a `helm install` of
a published chart with no further flags deploys exactly the image it was
packaged with.

### Upgrading from the pre-KVLab names

The project was renamed from `wekai` to `KVLab` with no compatibility shims.
Metrics are now `kvlab_router_*`, the pool label key is `kvlab.io/pool`, env
vars are `KVLAB_*`, and the config directory is `~/.kvlab`. Pod selectors are
immutable, so reinstall router releases (`helm uninstall` then `helm install`)
rather than `helm upgrade`, and update dashboards and alerts to the new metric
names.

## Building and publishing locally

| task | what it needs | what it does |
|---|---|---|
| `go test ./chart/` | helm only | renders the chart and asserts on the manifests — the fastest check, and enough for template changes |
| `task router:image` | docker | plain `docker build` of the router image |
| `task router:build` | a working Dagger engine | image **and** packaged chart, no push, no credentials — runs the release's own packaging code, so a chart checked here cannot differ from the published one |
| `task router:push` | Dagger + `QUAY_USERNAME`/`QUAY_PASSWORD` | publishes the image and the chart pinned to it; what the release workflow runs |

If Dagger cannot start — a locked-down Docker Desktop refusing to pull its engine
image is the usual cause — `go test ./chart/` and `task router:image` still cover
template and image correctness between them.

## Backend URLs and base paths

A backend is configured as a **base**; the client's request supplies the
**path**. For anything hosted at the root — vLLM, OpenAI, Anthropic — write the
bare host and nothing is added or stripped:

```
http://vllm:8000            +  POST /v1/chat/completions
  -> http://vllm:8000/v1/chat/completions
```

A base path exists for providers that are *not* at the root. Google's
OpenAI-compatible surface is one, and the prefix can only come from
configuration — without it the request lands on Google's native API, which
speaks a different protocol:

```
https://generativelanguage.googleapis.com/v1beta/openai  +  POST /v1/chat/completions
  -> https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
```

The base *replaces* the client's version segment rather than stacking on top of
it, which is what `base_url` means everywhere else it appears: an OpenAI
`base_url` ends in `/v1` and the caller appends `chat/completions`.

### The `/v1` suffix

Out of that same `base_url` habit, people often write:

```
http://vllm:8000/v1
```

Inference is unaffected — it composes back to exactly the same URL. But the
requests the router makes *on its own behalf* concatenate onto the base, so
`/v1/health`, `/v1/metrics` and `/v1/v1/models` all 404. The endpoint is then
taken for a hosted API, marked passive, and never actively health-checked — so
it serves fine until it dies and keeps looking healthy afterwards, while
contributing no models to `/v1/models` and no upstream metrics.

The router logs a warning at startup for any backend URL ending in `/v1`. Drop
the suffix; a genuine base path like `/v1beta/openai` is kept and used.

## Credentials

Two questions, kept separate: who may call the router, and how the router
authenticates to each pool.

### Serving only what you intend

**Setting an API key already does this.** A protected router serves the
dialect's own routes — `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, and
the rest of the table — all of them requiring the key, and nothing else. The
passthrough tier is closed, and so are the admin endpoints; ask for those
explicitly if you want them.

The list comes from the dialect's own route table, so claiming a path makes it
reachable on a protected listener without a second edit anywhere. What that does
NOT cover is a hosted provider's surface beyond inference — an Anthropic client
calling `/v1/organizations/*`, say. Those need `--path-allowlist` (`/v1/` admits
the subtree, and leaves the admin endpoints closed).

Setting a key says this listener faces users, and proxying arbitrary paths
through to a backend is not something a user-facing listener should do. The two
belong together, so one implies the other rather than being a second thing to
remember.

`--path-allowlist` overrides that default when you want a different set. Without
a key it is the only way to restrict anything: **empty means every path**, since
the passthrough tier is what lets one router front both a local fleet and a
hosted API on paths this dialect never claims.

```bash
kvlab router serve \
  --api-key-file /etc/kvlab/key \
  --path-allowlist /v1/chat/completions \
  --path-allowlist /v1/messages \
  --path-allowlist /v1/models \
  --backends pods:app=vllm
```

```yaml
router:
  pathAllowlist:
    - /v1/chat/completions
    - /v1/messages
    - /v1/models
```

Nothing outside the list is served — not the admin endpoints, and not a probe
path. There is no exemption. Matching is on segment boundaries, so
`/v1/mod` never admits `/v1/models`, and a trailing `/` denotes a subtree
(`/v1/responses/` admits `/v1/responses/abc`). A path off the list answers 404
rather than 401, so it does not confirm what the router has.

### Probes are on the metrics listener

`/liveness` and `/readiness` are served on `--metrics-listen`, not on the
serving port. Their answer is operational detail — how many backends exist, how
many are healthy, and why the router is not ready — and a user-facing router is
routinely unauthenticated, so on the serving port that detail would be public.

The chart probes the `metrics` port accordingly. A router started without
`--metrics-listen` serves no probes at all and says so at startup.

### Protecting the router

```bash
--api-key-file /etc/router/inbound-key    # preferred
--api-key <value>                         # visible in `ps` and in the pod spec
```

Every inference and admin request then needs the key. `/liveness` and
`/readiness` stay open so a kubelet needs no credential; `--require-auth-for-probes`
changes that.

### How a pool authenticates

A route says it in the same position as `as <model>`:

```
=> http://vllm:8000 using /etc/secrets/inner-key    the ROUTER's own key
=> https://api.anthropic.com using client           forward the CALLER's key
=> http://vllm:8000                                 send no credential
```

**Forwarding is opt-in and the default sends nothing.** A hosted API the user
pays for is the case that needs `using client` — the router has no key that
could work there. An internal backend must never receive a user's personal key,
so defaulting to forward would leak one on any route somebody forgot to
annotate. `--strip-auth-when` remains for dropping credentials by pattern.

`using <file>` wins over `using client`: a pool the router authenticates to
itself never also carries the caller's credential.

### Two routers: internal fleet behind a user-facing edge

The deployment this exists for. An inner router owns the fleet and requires a
key; an outer router is public, holds that key as a mounted secret, and sends
everything else to Anthropic with the user's own credential.

```bash
# inner — internal only, authenticated
kvlab router serve --listen :8080 \
  --api-key-file /etc/router/inbound-key \
  --backends 'http://vllm-a:8000|http://vllm-b:8000'

# outer — public, no inbound key
kvlab router serve --listen :8080 \
  --route 'llama,mistral => http://inner-router:8080 using /etc/router/inner-key' \
  --default 'https://api.anthropic.com using client'
```

A user calling the outer router for `llama-3` never learns the inner key; a
user calling it for `claude-*` pays with their own. Capture, if enabled, records
both.

```yaml
# outer values.yaml
router:
  routes:
    - patterns: [llama, mistral]
      endpoints: [http://inner-router:8080]
      using: /etc/router/inner-key
    - patterns: ["*"]
      endpoints: [https://api.anthropic.com]
      using: client
  secretMounts:
    - name: inner-key
      secretName: router-inner-key
      mountPath: /etc/router
```

## Per-user path prefixes

`--user-prefix` reads the FIRST path segment of every request as the caller's
name and removes it before anything else sees the path. A client sets

```
ANTHROPIC_BASE_URL=http://router:8080/alice
```

and its `/alice/v1/messages` is authorised, routed and forwarded as
`/v1/messages`, with `alice` recorded on the capture record. It is the only
handle available when the SDK will not send a header of your own.

Stripping happens **before the path allowlist, the route table and the upstream
path**, which is what makes per-user traffic ordinary traffic: it matches the
same dialect routes, gets the same prefix-cache affinity, and reaches a hosted
API on a path that API has heard of. Capture still records the path the client
sent, with the user as a field of its own.

**Every request is then expected to carry a prefix.** That is the contract
rather than a heuristic — there is no way to tell a user named `v1` from a
client that forgot its prefix — so a router running this way is dedicated to it.
Single-segment paths are left alone, so infra probes still work.

## Readiness

A router pod is **not ready until at least one endpoint is alive**. `/readiness`
returns 503 while no backend is healthy, so an orchestrator removes the pod from
its Service rather than sending it traffic it cannot serve. Backends are probed
immediately at startup and then on `--health-interval`, so readiness reflects the
fleet rather than the router's own process being up — that is `/liveness`.

Probe the router on `/readiness` and `/liveness` (`/healthz` and `/livez` are
accepted aliases). **Do not probe an arbitrary path**: any path the dialect does
not claim is proxied to a backend, so a probe on the wrong path is answered by
something that knows nothing about it.

A saturated router stays ready. It sheds with 429 rather than going NotReady,
because it is working, not broken.

## Observability

`/router-viz` and the probes are on `--metrics-listen`, never on the serving
path. `/metrics` is there too — and, **when no API key is set**, also on the
serving port.

That second mount exists so a client can derive its scrape target from the
serving endpoint (strip `/v1`, append `/metrics`) instead of being told a second
address, because being told a second address is a step that gets skipped and the
run then records nothing while looking healthy. An API key is the production
shape; a keyless router is a dev or benchmark deployment, where the serving
listener is already unauthenticated.

It is not a passthrough. An earlier version proxied the scrape to one backend
and returned that backend's `vllm:*` counters as though they were the fleet's —
wrong by a factor of the fleet size, and moving backwards whenever consecutive
scrapes landed on different backends. Both mounts now serve the same aggregate,
from the same handler value.

Upstream vLLM counters are aggregated **by default**, independently of the API
key — the key decides where `/metrics` is exposed, not whether the counters
exist. `--no-vllm-metrics` (alias `--no-backend-metrics`) turns it off. Only
endpoints discovered to be vLLM or SGLang are scraped, so a hosted API in the
same router is never asked. The tracked counters are
`vllm:prompt_tokens_by_source_total`, `sglang:cached_tokens_total` and
`sglang:prompt_tokens_total`, re-emitted under the same names (so a benchmark's
`--sglang-metrics-url` can point at the router's metrics listener); per-rank
SGLang gauges are deliberately not aggregated. `--backend-metrics-name` /
`--backend-metrics-interval` are engine-neutral aliases of the `--vllm-metrics-*`
flags.

Totals accumulate **deltas**, so they never rewind when a pod restarts or the
fleet scales down, and a backend that cannot be reached simply stops
contributing rather than failing the cycle. That is why it needs no opt-in:
there is no failure the operator has to protect against by staying silent.

The cost is that flat totals mean two opposite things — an idle fleet and an
unreachable one produce the same numbers — and telling them apart takes two
metrics, not one.

`router_vllm_metrics_endpoints` covers the case where a backend is still in the
pool and cannot be scraped: `contributing` falls while `asked` holds. It does
**not** cover a backend leaving. `asked` tracks the live discovered set, so a
deleted pod is out of it before a scrape against it can fail — both labels drop
together and the ratio reads 1.0 while the fleet is down a node. Measured on
hardware: killing a pod took the gauge 8/8 → 7/7, never 8/7.

So alert on all three:

| symptom | rule |
|---|---|
| a backend is present but unscrapeable | `contributing < asked` |
| the fleet shrank | `router_backends_total{state="healthy"}` fell |
| aggregation has nothing at all | `asked == 0` |

| metric | what it tells you |
|---|---|
| `router_route_decisions_total` | which tier decided: `cache`, `split`, `overflow`, `load`. Aggregate by label. |
| `router_cache_avg_copies` | mean backends holding each block; ~1.0 means no duplication |
| `router_cache_guard_rejects_total` | 429s the split guard caused. **Read with `avg_copies`** — a misconfiguration lands in one or the other depending on where the guard sits, and either alone misses half the failure space |
| `router_signal_fired_total` | which capacity signal is actually driving decisions |
| `router_cache_overflows_total` | requests `--transient-fallback-threshold` served without marking a holder. **Read with `guard_rejects`** — same situation, opposite outcomes; the ratio is what the threshold buys |
| `router_cache_soft_blocked_total` | decisions where every available holder was past `--soft-node-concurrency`. The **trigger**, not an outcome: flat at zero means the soft limit is too high to bind, equal to the decision count means it is too low and the fleet lives above it |
| `router_cache_stretches_total` | requests kept on a holder already past the soft limit because nothing cleared the guard. `soft_blocked − stretches` is how often spreading actually worked |
| `router_cache_stretch_inflight` | in-flight on the chosen holder at selection time, stretch path only. Says whether the soft→hard band is entered lightly or is where the fleet lives |
| `router_retries_total{reason="capacity_saturated"}` | waits caused by every backend being full. The transient fallback cannot apply here — there was no candidate — so waiting is the only move |
| `router_retries_total{reason="capacity_guard_blocked"}` | waits caused by the split guard. The fallback is tried BEFORE this error is returned, so a count here with `overflows_total` at zero means the threshold is too tight or off — not that the router waited instead of falling back |
| `router_retry_wait_seconds` | latency `--retry-time-limit` added, **per request** — `_count{outcome="satisfied"}` is how many requests the waiting rescued, `{outcome="expired"}` how many spent the budget and got a 429 anyway, and the quantiles what it cost. Spans the first refusal to the *start* of the attempt that ended the wait, so it excludes that attempt's service time and stays bounded by the budget in every outcome — quantiles are comparable across them. End-to-end cost is `router_request_duration_seconds`. `retries_total` counts *attempts*, so it answers neither |
| `router_prefill_rejections_total` | `large_prefill_rejected` 503s the router absorbed (`--prefill-split` only, like the rest of this family) |
| `router_prefill_jobs_total{outcome,reason}` | prefill jobs. `ok` (no reason); `failed`, by `reason`: `timeout` (`--request-timeout` on the job's request to the node, counted from when it is sent, not from when it was reserved), `status_4xx` / `status_5xx`, `transport` (connection failed or broke), `rejected` (refused as a large prefill despite the bypass), `other` — the request is then sent on with the bypass; `cancelled` by the caller, `reason` `abandoned` (it left while the job waited in the router for a slot) or `client_cancelled` (while the job ran or waited out a 429) — nothing more is sent for it, and the job's slot, budget share and pending tokens are returned. A caller leaving used to count as `failed` (and took the bypass fallback); it no longer does. Each failure logs `prefill job failed` (warn), each cancellation `prefill job cancelled` (info), with the reason, backend, tokens, time waited for a slot and time running, status and error; at most one line per outcome and reason every 10s, the next one carrying `suppressed_since_last` |
| `router_prefill_conversions_total` / `router_prefill_releases_total` | decode nodes taken out for prefill, and returned to decode as soon as their last job finishes |
| `router_prefill_bypass_fallbacks_total{reason}` | requests served with the bypass and no prefill: `no_node` (none could be added) or `job_failed` |
| `router_prefill_nodes{mode}` / `router_prefill_node_mode{backend}` | backends per mode, and each backend's mode (0 decode, 1 draining, 2 prefill); under `--prefill-split-mode=pd` these, and every other `router_prefill_*` series, describe the dynamic P set |
| `router_prefill_queued_jobs` / `router_prefill_job_retries_total` | prefill jobs waiting for a slot under `--max-node-concurrency` (a prefill node never runs more jobs than that, counting its other in-flight requests), and jobs retried on the same node after it answered 429 |
| `router_prefill_pending_tokens{backend}` | queued plus in-flight prefill tokens on a backend. A node is sent at most `--prefill-node-budget-tokens` of it at once; the rest waits in the router (`router_prefill_queued_jobs`) and may move. `/router-viz` shows each node's `running_tokens` |
| `router_prefill_fallbacks_skipped_total` | rejected requests whose bypass fallback (no prefill node, or a job that did not succeed) was not sent because the caller had already left; not in `router_prefill_bypass_fallbacks_total` |
| `router_prefill_job_moves_total` | queued prefill jobs moved to another prefill node that could start them at once |
| `router_pd_prefill_legs_total{outcome}` | prefill legs sent to a P node (`--pd-prefill-backends` only): `ok`, `failed` (non-2xx or transport error) or `no_params` (answered without `kv_transfer_params`) |
| `router_pd_fallbacks_total{reason}` | requests sent to a D node without `kv_transfer_params`, so it recomputes: `no_prefill_node`, `prefill_failed` or `no_params` |
| `router_pd_prefill_retries_total` | prefill legs retried after every P node was saturated or the chosen one answered 429 |
| `router_prefill_job_tokens` / `router_prefill_job_wait_seconds` | per prefill job (both split modes): the tokens it is charged against the budget, and its router wait from placement to dispatch |
| `router_prefill_drains_deferred_total` | jobs that found a drain due but held back by `--prefill-drain-min-interval` |
| `router_pd_prefill_held_tokens{tree}` | `--prefill-split-mode=pd`, per prefill leg: the deepest prefix of its prompt the router's tree records, `tree="prefill"` (any P node) or `tree="decode"` (any D node); the leg is charged its prompt less the larger, the decode depth cut to `--prefill-store-unit-tokens` |
| `router_pd_prefill_service_seconds` | per successful prefill leg: from sending it to a P node to its `kv_transfer_params` |
| `router_pd_leg_duration_seconds{leg}` | wall time of the `prefill` leg (retries included) and the `decode` leg (to the end of its response) |
| `router_backend_queued` | requests currently waiting in ONE backend's `--max-node-concurrency` queue. Non-zero only while `--retry-time-limit` is set; the per-backend companion to the fleet-wide `router_backend_queue_wait_seconds` below |
| `router_backend_queue_wait_seconds` | latency spent in a backend's wait queue, by outcome — `woken` (a slot freed, request re-decided), `expired` (budget ran out, same 429 as the poll path), `abandoned` (caller disconnected). Separate from `router_retry_wait_seconds`: that one measures the whole-fleet poll loop, this one a wait bound to a single backend, and the two describe different mechanisms for the same refusal |
| `router_backend_queue_full_total{backend}` | requests refused a place in line outright because that backend's queue was already at its depth bound (`--max-node-concurrency`). Answered `503`, not `429` — the request never waited at all |
| `router_vllm_metrics_endpoints` | upstream endpoints `asked` vs `contributing` on the last aggregation cycle. Catches a backend that is **in the pool and unscrapeable** — hung vLLM, partition, present-but-unready — which is otherwise invisible in flat `vllm:` totals. It does **not** catch a backend leaving: `asked` tracks the live set, so a deleted pod drops both labels together and the ratio still reads 1.0. Use `router_backends_total` for that, and alert on `asked == 0` separately |
| `router_stream_aborted_total{reason="upstream_error"}` | streams the BACKEND stopped sending before the terminal marker. Feeds the circuit breaker, so a backend truncating responses is ejected rather than counted healthy |
| `router_stream_aborted_total{reason="client_disconnect"}` | streams the CALLER abandoned. Does **not** count against the backend — on a saturated fleet callers give up constantly, and blaming them would eject healthy backends exactly when losing one costs most |
| `router_cache_tree_runs` / `_tail_set` | tree size, for memory |

Every series above with a closed set of label values exists **at 0 from
startup** — the retry reasons and outcomes, every `route_decisions` tier, every
`signal_fired` signal, and each pool's cache counters. Absent and zero are
different claims: a scrape missing `router_retries_total` is equally consistent
with a budget that was never needed and one that was never wired up, so a
missing series must never be the way a router says "this did not happen".

Label sets with no complete list — backend URLs, upstream error kinds — stay
lazy, since enumerating them would mean inventing backends that do not exist.

Every cache and signal metric carries a **`pool`** label. A router may front
several pools whose caches are unrelated, so summing across them describes
nothing — aggregate by pool.

`--vllm-metrics` additionally aggregates upstream vLLM counters into
router-level totals served alongside the router's own. Only endpoints discovered
to be vLLM are scraped, and totals accumulate **deltas**, so they never rewind
when a pod restarts or the fleet scales down. A counter that decreases silently
breaks `rate()` and `increase()`.

## Capture

`--capture raw|redacted` (off by default) records every proxied exchange to JSONL for later
analysis or replay. `redacted` keeps structured metadata — block hashes, token
counts, tool-use ids — without bodies. This is how the replay corpora used for
benchmarking are collected.

## Two 429s, and what they mean

- **`all_backends_saturated`** — nothing could take work.
- **`split_guard_blocked`** — capacity existed, and the guard refused to spend it
  on a duplicate copy of the prefix. Not an error: the policy working as
  designed.

Both carry `Retry-After: 1`. **Clients must retry.** A client treating 429 as
fatal will see throughput collapse under load; the kvlab replay client backs off
10ms doubling to a 3s cap with a 30s budget.

## Testing without hardware

See [`.ainav/router-testing/`](../.ainav/router-testing/index.md). `mock-vllm`
is a GPU-less vLLM with three surfaces (`--surface vllm|anthropic|hosted`) so
every deployment shape above can be exercised offline, including a vLLM fronted
with an Anthropic API. For the translation mode, a chat request can carry an
`X-Mock-Script` header scripting the reply (reasoning, text, streamed and
interleaved tool calls, finish/stop reasons, a cut-off stream, an error), and
`mock-vllm --agentic` answers tool-offering requests like a tool-using
assistant so a client can run whole loops — see
[`.ainav/router-testing/anthropic-api.md`](../.ainav/router-testing/anthropic-api.md).
