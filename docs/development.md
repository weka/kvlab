# Development: embedding and publishing

## Embedding

Applications that embed kvlab's command groups directly (rather than running the
standalone binary) have two extension points, both in `cli/`:

- `cli.SetGlobalOptions(*GlobalOptions)` for binaries that parse their own
  `cli.GlobalOptions` before calling `flags.Parse` (this is what `main.go` does).
  go-flags mutates that struct in place, so by the time any command runs, kvlab's
  internal `config.Config` is already in sync.
- `cli.PreExecute func(ctx context.Context) error` for applications with their
  own richer global-options type that do not construct a `cli.GlobalOptions` at
  all. Set it once; it runs at the top of every embedded command's `Execute()`,
  before the command touches `config.Config`. Use it to copy the embedder's own
  parsed flags into `config.Config` and to register model-registry hooks. This
  lets `BenchmarkCommands`/`RouterCommands`/`EvalCommands` be embedded as plain
  type aliases with no per-command wrapping.

An embedder with its own named models should also set `llm.ResolveModel` and
`llm.LookupModelByIdentifier`; otherwise only dynamic and openrouter specs
resolve.

## Layout

- `llm/` — raw LLM client layer (Anthropic, OpenAI, OpenAI Responses, Gemini
  native), dynamic model-spec parsing, cost calculation, mock servers for tests.
- `tools/` — base tool/toolset types shared by the LLM client layer and the
  benchmark tool-chain eval.
- `benchmark/` — benchmark and replay engine.
- `cli/` — go-flags command trees for `benchmark`, `router`, `eval`.
- `config/` — minimal global config and environment-based API key loading.

## Replay image

`Dockerfile` builds a self-contained image: a golang stage compiles the `kvlab`
binary, and a second `COPY --link --from=` stage embeds one replay JSONL at
`/kvlab/replay.jsonl`, pulled from a separately published scratch image (the
`REPLAY_IMAGE` build arg; default
`quay.io/weka.io/kvlab:replay-40d-oct14-337a03810637`). Replay artifacts and the
app image share the kvlab quay repo, told apart by the `replay-` tag prefix. To
fetch an artifact standalone, see *Downloading captures* in the README. Build
locally with:

```
task docker:build   # override REPLAY_IMAGE=... to embed a different capture
```

## CI / Publishing

Publishing is a self-contained Dagger module (Python SDK, engine pinned to
`v0.21.7`) rooted at `dagger.json` / `.dagger/src/kvlab_core_flows/`. All Go
dependencies needed to build the image are public, so no SSH socket forwarding or
registry credentials are required for the image build itself.

Three functions:

- `push-replay` publishes a replay JSONL as a minimal scratch image to
  `quay.io/weka.io/kvlab` by default (registry overridable), alongside the app
  image. The tag is `replay-<label>-<sha12>` with a `LABEL`, and `replay-<sha12>`
  without one; `sha12` is the first 12 hex characters of the file's sha256, so the
  hash stays in the tag whether or not a label is given.
- `publish` builds the repo's `Dockerfile` via `Directory.docker_build()` (the
  Dockerfile is the single source of truth for the binary and runtime; the module
  does not reimplement its steps), tags it `v999.0.0-<sha12>` (sha12 of the source
  directory's content digest), and publishes to `quay.io/weka.io/kvlab` by
  default. It builds the Dockerfile's `base` stage plus the binary, and
  publishes it on top of the replay image named by `--replay-image` (the
  Dockerfile's `REPLAY_IMAGE`) without ever downloading that image's layers:
  `.dagger/graft` writes each platform's manifest as the replay image's layer
  (by digest, so the registry is only asked whether it has it) followed by the
  runtime's, with the runtime's config. `/kvlab/replay.jsonl` is a symlink to
  the replay image's `/replay.jsonl`. So a release neither downloads,
  recompresses nor re-uploads the artifact, and a node that pulled any earlier
  release does not download or unpack it again. The binary comes from
  `--binaries` when given (the release passes the ones it built for the GitHub
  Release); otherwise the Dockerfile's builder stage cross-compiles it on the
  engine's own platform, never under emulation. The push takes
  `--registry-username`/`--registry-password` (Dagger secrets), since it runs
  in a container without the caller's docker login.
- `push-helm` first runs the exact same image build and publish as `publish`
  (shared internal helper, not a separate code path), then packages `chart/kvlab`
  and pushes it to an OCI Helm registry (`quay.io/weka.io/helm` by default).
  Version pinning is pure propagation: only `Chart.yaml`'s `version`/`appVersion`
  are stamped, in lockstep with the image just pushed, and the deployment template
  resolves the image tag via `imageTag | default .Chart.AppVersion`; the packaged
  `values.yaml` carries no hardcoded version (`imageTag` stays `""`, and only
  `imageRepository` is synced to the push registry). A `helm install` of a chart
  version with no further `--set` flags always deploys the image published under
  that same version. Because the image publish is awaited before any chart
  packaging, "image pushed before chart push" holds by construction. Takes
  `helm-username`/`helm-password` as Dagger secrets.

```
task replay:push REPLAY=/path/to/replay.jsonl LABEL=40d-oct14   # dagger call push-replay
task app:push                                    # dagger call publish (image only; QUAY_USERNAME/QUAY_PASSWORD)
task helm:push                                   # dagger call push-helm (image + chart, image first)
```

`helm:push` reads Helm registry credentials from the `QUAY_USERNAME` and
`QUAY_PASSWORD` environment variables.

`.dagger/sdk/` (the generated Dagger client bindings) is gitignored: run
`dagger develop` once after a fresh clone (or let the first `dagger call` /
`task app:push` do it implicitly) to regenerate it locally. Nothing under `sdk/`
is hand-edited or committed.

Releases from `main` are cut by `.github/workflows/release.yml`: the next semver
comes from Conventional Commits (see `CLAUDE.md`), `dagger call release` runs
`push-helm` and `push-router-helm` concurrently in one engine session, publishing
both images and charts under that `vX.Y.Z` instead of the content-hash stamp, and the GitHub
Release carries per-version install commands.
