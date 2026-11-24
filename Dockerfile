# syntax=docker/dockerfile:1.26@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
#
# kvlab replay image: the kvlab binary (benchmark/router/eval
# command trees) plus one embedded router-replay JSONL artifact, so a
# deployment can run `kvlab benchmark auto --router-replay-file
# /kvlab/replay.jsonl ...` with no external volume or fetch step.
#
# The replay artifact is published separately (`task replay:push` /
# `dagger call push-replay`) as a minimal scratch image in the same kvlab
# quay repo, distinguished by the replay- tag prefix.

# ARGs used in a FROM must be declared before the first FROM (global scope).
# Override with --build-arg REPLAY_IMAGE=... to embed a different capture.
ARG REPLAY_IMAGE=quay.io/weka.io/kvlab:replay-40d-oct14-337a03810637

# ---- Builder: compile the kvlab binary ----
# The builder runs on the BUILD host's platform and cross-compiles for the
# target: kvlab is pure Go (CGO_ENABLED=0), so GOOS/GOARCH are all a target
# needs. Building under the target platform instead runs the whole Go
# toolchain under QEMU for a foreign architecture, which is several times
# slower and has crashed `go tool compile` outright. (Dagger's docker_build
# ignores this --platform; the release builds this stage for the engine's own
# platform with TARGETOS/TARGETARCH passed as build args instead.)
FROM --platform=$BUILDPLATFORM golang:1.25.7-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# TARGETOS/TARGETARCH are set by buildkit from the platform being built, so one
# Dockerfile produces every architecture in the manifest list. Hardcoding
# amd64 here silently pinned the binary even when the container platform said
# otherwise — an arm64 node would have pulled an arm64 image containing an
# amd64 executable, which fails at exec with no useful message.
ARG TARGETOS TARGETARCH
ENV CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH
# -trimpath: the release builds the GitHub Release binaries with these same
# flags and puts those binaries in the image (see .github/workflows/release.yml).
RUN go build -trimpath -o /out/kvlab .

# ---- Replay artifact source ----
# Pinned to amd64 deliberately: this stage contributes only replay.jsonl, a
# platform-independent data file, and the replay image is published as a single
# amd64 manifest. Without the pin an arm64 build would look for an arm64 variant
# that does not exist and fail the whole build for a file with no architecture.
FROM --platform=linux/amd64 ${REPLAY_IMAGE} AS replay

# ---- Base: the runtime image minus the binary and the replay artifact ----
# A stage of its own so the release can build exactly this and add the binary
# it cross-compiled (see .dagger): Dagger's docker_build sets BUILDPLATFORM to
# the target platform, so the builder stage above would run under QEMU there.
FROM alpine:latest AS base
LABEL org.opencontainers.image.title="kvlab" \
      org.opencontainers.image.description="LLM benchmarking, router/proxy, and capture-replay toolkit" \
      org.opencontainers.image.source="https://github.com/weka/kvlab"
RUN apk add --no-cache ca-certificates
ENTRYPOINT ["/usr/local/bin/kvlab"]

# ---- Runtime + replay artifact (local `docker build`) ----
# --link makes each COPY an independent layer that does not depend on the
# layers below it, so under BuildKit a Go change does not recopy the artifact.
# The release does not build this stage: a COPY still produces a NEW layer
# blob, which has to be compressed (1.3 GB gzip) and uploaded for each
# architecture on every release. The release instead lays `base` + binary over
# the replay image's own layer, byte for byte, so the registry already holds it
# and a node that pulled an earlier release already has it unpacked.
FROM base
COPY --link --from=builder /out/kvlab /usr/local/bin/kvlab
COPY --link --from=replay /replay.jsonl /kvlab/replay.jsonl
