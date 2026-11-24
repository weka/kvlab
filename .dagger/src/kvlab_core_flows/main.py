import asyncio
import re
import time
from typing import Annotated

import dagger
from dagger import BuildArg, Ignore, dag, function, object_type

LINUX_AMD64 = dagger.Platform("linux/amd64")

# Every published image is a manifest list covering both architectures, so one
# tag installs on an x86 fleet and on an arm workstation without the chart or
# the operator choosing. A single-arch image does not fail at pull time on the
# wrong node — it fails at exec, with no message that names the cause.
#
# The cost is a second build per publish. It is paid because the alternative is
# an image that works everywhere it has been tried and nowhere it has not.
PUBLISH_PLATFORMS = [
    dagger.Platform("linux/amd64"),
    dagger.Platform("linux/arm64"),
]

# Mirrors kvlab's own .gitignore so a publish digest isn't polluted by
# local build artifacts, IDE state, or the dagger module's own generated
# sdk/. Leading "/" anchors to the repo root — an unanchored "kvlab"
# would also match chart/kvlab (see .gitignore history).
SOURCE_IGNORE = [
    "*/.git",
    "*/.DS_Store",
    ".dagger",
    "!.dagger/src/",
    ".idea",
    ".temp",
    "/bin",
    "/dist",
    "/kvlab",
    "/kvlab-core",
    "*.log",
    "*.out",
    "/results",
]

CHART_DIR = "chart/kvlab"
CHART_NAME = "kvlab"

# The router ships as a SECOND image and chart from the same commit. It is the
# same kvlab binary; what differs is that the router image carries no embedded
# replay artifact. That artifact is multi-GB and exists so a benchmark pod can
# replay captured traffic without a volume — a router never opens it, and making
# every router replica pull it would be pure cost. Two Dockerfiles rather than a
# build arg, so a router image that accidentally includes it is impossible
# rather than merely unlikely.
BENCHMARK_DOCKERFILE = "Dockerfile"
ROUTER_DOCKERFILE = "Dockerfile.router"
ROUTER_CHART_DIR = "chart/router"
ROUTER_CHART_NAME = "kvlab-router"


# Where the benchmark image exposes the replay artifact. The chart's default
# replayFile and every documented command line use this path.
REPLAY_PATH = "kvlab/replay.jsonl"
BINARY_PATH = "/usr/local/bin/kvlab"


# The Go image the graft tool is built with; same as the Dockerfiles' builder.
GO_IMAGE = "golang:1.25.7-alpine"


async def _image(
    source: dagger.Directory,
    dockerfile: str,
    platform: dagger.Platform,
    binaries: dagger.Directory | None = None,
    binary_name: str = "kvlab",
) -> dagger.Container:
    """One architecture of an image: the Dockerfile's `base` stage for
    `platform`, plus the binary.

    The binary is `binaries/<binary_name>_<os>_<arch>/kvlab` when `binaries` is
    given (the release passes the ones it already built, with the Dockerfile's
    flags, so nothing compiles twice). Otherwise the Dockerfile's `builder`
    stage cross-compiles it on the engine's own platform: Dagger's docker_build
    sets BUILDPLATFORM to the target, so `FROM --platform=$BUILDPLATFORM` does
    not keep the builder native, and compiling for arm64 under QEMU took ~16
    minutes per image and has segfaulted `go tool compile`. The base stage
    still builds for the target, which only runs `apk add`.
    """
    target_os, target_arch = str(platform).split("/")[:2]
    if binaries is not None:
        binary = binaries.file(f"{binary_name}_{target_os}_{target_arch}/kvlab")
    else:
        binary = source.docker_build(
            platform=await dag.default_platform(),
            dockerfile=dockerfile,
            target="builder",
            build_args=[
                BuildArg(name="TARGETOS", value=target_os),
                BuildArg(name="TARGETARCH", value=target_arch),
            ],
        ).file("/out/kvlab")
    return (
        source.docker_build(platform=platform, dockerfile=dockerfile, target="base")
        .with_file(BINARY_PATH, binary, owner="0:0", permissions=0o755)
    )


async def _publish_onto(
    variants: list[dagger.Container],
    base_image: str,
    image_name: str,
    username: dagger.Secret | None,
    password: dagger.Secret | None,
    insecure: bool,
) -> None:
    """Publishes `variants` as one multi-arch image whose bottom layers are
    `base_image`'s, without the engine ever fetching those layers.

    The variants are exported as an OCI layout (a few tens of MB) and the graft
    tool (.dagger/graft) writes, per platform, a manifest listing the base
    image's layers by digest followed by the variant's, with the variant's
    config. The registry is only asked whether it has the base layers; their
    blobs are never downloaded or uploaded. Pulling a 1.3 GB base into a fresh
    CI engine just to push it back took ~6 of a release's minutes.
    """
    layout = dag.container().as_tarball(platform_variants=variants)
    ctr = (
        dag.container()
        .from_(GO_IMAGE)
        .with_mounted_cache("/go/pkg/mod", dag.cache_volume("kvlab-go-mod"))
        .with_mounted_cache("/root/.cache/go-build", dag.cache_volume("kvlab-go-build"))
        .with_directory("/graft", dag.current_module().source().directory("graft"))
        .with_workdir("/graft")
        .with_exec(["go", "build", "-o", "/usr/local/bin/graft", "."])
        .with_file("/layout.tar", layout)
        .with_exec(["sh", "-c", "mkdir /layout && tar -xf /layout.tar -C /layout"])
    )
    if username is not None and password is not None:
        ctr = (
            ctr.with_secret_variable("GRAFT_USERNAME", username)
            .with_secret_variable("GRAFT_PASSWORD", password)
        )
    args = ["graft", "-layout", "/layout", "-base", base_image, "-dest", image_name]
    if insecure:
        args.append("-insecure")
    # A publish is a side effect: never let the exec cache skip it.
    await ctr.with_env_variable("GRAFT_RUN", str(time.time_ns())).with_exec(args).sync()


def _benchmark_variant(image: dagger.Container) -> dagger.Container:
    """One architecture of the benchmark image, minus the replay layer that
    `_publish_onto` puts underneath it.

    The replay layer is the image's BOTTOM layer and is the replay image's blob
    itself, not a copy of the file. A copy (the Dockerfile's final COPY) is a
    new 1.3 GB gzip blob per architecture per release: minutes to compress,
    2.6 GB to upload, and 9.5 GB for every node to download and unpack again on
    each upgrade. Reusing the blob makes all three free after the first time:
    the registry already holds it, and a node that ran any earlier release
    already has it unpacked, because a bottom layer's identity does not depend
    on anything above it.

    The replay image keeps the file at /replay.jsonl; /kvlab/replay.jsonl is a
    symlink to it, so the documented path is unchanged. The replay image is
    amd64-only, which does not matter: the layer is a data file, and each
    variant's platform and config are its own.
    """
    return image.with_rootfs(image.rootfs().with_symlink("/replay.jsonl", REPLAY_PATH))


async def _calc_version(src: dagger.Directory) -> str:
    """Content-hash dev version (v999.0.0-<sha12>) of the source tree."""
    digest = await src.digest()
    sha = digest.split(":")[-1]
    version = f"v999.0.0-{sha[:12]}"
    return version


async def _publish_image(
    source: dagger.Directory,
    registry: str,
    replay_image: str,
    version: str = "",
    binaries: dagger.Directory | None = None,
    username: dagger.Secret | None = None,
    password: dagger.Secret | None = None,
    insecure: bool = False,
) -> tuple[str, str]:
    """Builds and publishes the kvlab image from this repo's own
    Dockerfile. Shared by `publish` and `push_helm` so there is exactly one
    build/tag/publish code path — `push_helm` awaits this to completion
    before touching the chart, which is what makes "image pushed before
    chart push" hold by construction rather than by convention.

    Returns (image_name, version).

    version: explicit version stamp (e.g. a semver tag from CI); when empty,
    falls back to the content-hash scheme (_calc_version's v999.0.0-<sha12>).
    username/password: registry credentials for the push (the graft tool runs
    in a container, without the caller's docker login).
    """
    if not version:
        version = await _calc_version(source)
    variants = [
        _benchmark_variant(await _image(source, BENCHMARK_DOCKERFILE, p, binaries, "kvlab"))
        for p in PUBLISH_PLATFORMS
    ]
    image_name = f"{registry}:{version}"
    await _publish_onto(variants, replay_image, image_name, username, password, insecure)
    return image_name, version


async def _publish_router_image(
    source: dagger.Directory,
    registry: str,
    version: str = "",
    binaries: dagger.Directory | None = None,
) -> tuple[str, str]:
    """Builds and publishes the replay-less router image.

    Same source, same binary, different Dockerfile — see ROUTER_DOCKERFILE.

    Returns (image_name, version).
    """
    if not version:
        version = await _calc_version(source)
    variants = [
        await _image(source, ROUTER_DOCKERFILE, p, binaries, "kvlab-router")
        for p in PUBLISH_PLATFORMS
    ]
    image_name = f"{registry}:{version}"
    # Published from an empty container so no single architecture is the
    # "real" one and the others attachments — every variant is equal in the
    # resulting manifest list.
    await dag.container().publish(image_name, platform_variants=variants)
    return image_name, version


def _package_chart(
    chart: dagger.Directory,
    registry: str,
    version: str,
) -> dagger.Container:
    """Packages one chart pinned to one image, WITHOUT pushing.

    Split from pushing so a local build can exercise the identical packaging —
    the stamping below IS the pinning contract, and a local path that
    reimplemented it with its own `helm package` would drift from the released
    artifact precisely where it matters.

    Version pinning is pure propagation: only Chart.yaml is stamped, and the
    deployment template resolves the tag via `imageTag | default
    .Chart.AppVersion`, so values.yaml never carries a hardcoded version.
    imageRepository IS synced to the real destination, so publishing to a
    custom registry cannot yield a chart pointing at the default one.
    """
    return (
        dag.container(platform=LINUX_AMD64)
        .from_("alpine:latest")
        .with_exec(["apk", "add", "--no-cache", "helm"])
        .with_directory("/chart", chart)
        .with_exec(["sed", "-i", f"s/^version:.*/version: {version}/", "/chart/Chart.yaml"])
        .with_exec(["sed", "-i", f's/^appVersion:.*/appVersion: "{version}"/', "/chart/Chart.yaml"])
        .with_exec(["sed", "-i", f"s|^imageRepository:.*|imageRepository: {registry}|", "/chart/values.yaml"])
        .with_exec(["helm", "package", "/chart", "--destination", "/out"])
    )


async def _package_and_push_chart(
    chart: dagger.Directory,
    chart_name: str,
    registry: str,
    version: str,
    helm_registry: str,
    helm_username: dagger.Secret,
    helm_password: dagger.Secret,
) -> str:
    """Packages a chart and pushes it. See _package_chart for the pinning."""
    registry_host = helm_registry.split("/")[0]
    packaged = _package_chart(chart, registry, version)
    await (
        packaged
        .with_secret_variable("HELM_USER", helm_username)
        .with_secret_variable("HELM_PASS", helm_password)
        .with_exec(["sh", "-c",
                    f"echo $HELM_PASS | helm registry login {registry_host} -u $HELM_USER --password-stdin && "
                    f"helm push /out/{chart_name}-*.tgz oci://{helm_registry}"])
    ).stdout()
    return f"oci://{helm_registry}/{chart_name}:{version}"


@object_type
class KvlabCoreFlows:
    @function
    async def push_replay(
        self,
        replay: dagger.File,
        registry: str = "quay.io/weka.io/kvlab",
        label: str = "",
    ) -> str:
        """Publishes a router replay file as a minimal scratch image.

        Ported from kvlab's .dagger push_replay: identical tag scheme
        (replay-<sha12 of content>). Replay artifacts live in the same
        kvlab quay repo as the app image, distinguished by the replay-
        tag prefix.
        The image carries the replay JSONL at /replay.jsonl, so this repo's
        own Dockerfile embeds it via:
        COPY --link --from=<registry>:replay-<sha12> /replay.jsonl /kvlab/replay.jsonl

        Args:
            replay: The local replay JSONL file to publish (mandatory).
            label: Optional human-readable part of the tag, e.g. "40d-oct14",
                giving replay-<label>-<sha12>. The hash stays, so the tag
                still names the exact content.
        """
        if label and not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,60}", label):
            raise ValueError(
                f"label {label!r}: use lowercase letters, digits, '.', '_' or '-'"
            )
        sha_line = await (
            dag.container(platform=LINUX_AMD64)
            .from_("alpine:latest")
            .with_file("/replay.jsonl", replay)
            .with_exec(["sha256sum", "/replay.jsonl"])
            .stdout()
        )
        sha = sha_line.split()[0]
        tag = f"replay-{label}-{sha[:12]}" if label else f"replay-{sha[:12]}"

        replay_image = dag.container(platform=LINUX_AMD64).with_file(
            "/replay.jsonl", replay
        )
        image_name = f"{registry}:{tag}"
        await replay_image.publish(image_name)

        return (
            f"Published replay image: {image_name}\n"
            f"Replay file path in image: /replay.jsonl\n"
            f"sha256: {sha}"
        )

    @function
    async def publish(
        self,
        source: Annotated[dagger.Directory, Ignore(SOURCE_IGNORE)],
        registry: str = "quay.io/weka.io/kvlab",
        replay_image: str = "quay.io/weka.io/kvlab:replay-40d-oct14-337a03810637",
        version: str = "",
        registry_username: dagger.Secret | None = None,
        registry_password: dagger.Secret | None = None,
        insecure_registry: bool = False,
        binaries: dagger.Directory | None = None,
    ) -> str:
        """Builds and publishes the kvlab image from this repo's own Dockerfile.

        Uses Directory.docker_build() against the existing Dockerfile at the
        repo root — the Dockerfile is the single source of truth for the
        build; this function does not reimplement its steps. Tagged with
        the same version stamp scheme kvlab uses (_calc_version:
        v999.0.0-<sha12> of the source directory's digest) unless an
        explicit version is given.

        Args:
            registry: Target image registry:repo (tag is appended).
            replay_image: The replay image whose layer becomes the image's
                bottom layer (the same capture the Dockerfile's REPLAY_IMAGE
                build-arg selects for a local `docker build`).
            version: Explicit version stamp (e.g. a semver tag from the
                release workflow). Empty = content-hash scheme.
            registry_username: Registry credentials for the push.
            registry_password: Registry credentials for the push.
            insecure_registry: Allow a plain-HTTP registry (local testing).
            binaries: Prebuilt binaries, laid out as
                <name>_<os>_<arch>/kvlab (the release passes the ones it built
                for the GitHub Release). Omitted = compile from the Dockerfile.
        """
        image_name, _version = await _publish_image(
            source, registry, replay_image, version, binaries,
            registry_username, registry_password, insecure_registry,
        )
        return f"Published kvlab image: {image_name}"

    @function
    async def push_helm(
        self,
        source: Annotated[dagger.Directory, Ignore(SOURCE_IGNORE)],
        helm_username: dagger.Secret,
        helm_password: dagger.Secret,
        registry: str = "quay.io/weka.io/kvlab",
        helm_registry: str = "quay.io/weka.io/helm",
        replay_image: str = "quay.io/weka.io/kvlab:replay-40d-oct14-337a03810637",
        version: str = "",
        binaries: dagger.Directory | None = None,
    ) -> str:
        """Publishes the kvlab image, then packages and pushes
        chart/kvlab to an OCI Helm registry with the chart's image
        reference pinned to that exact just-published image — a
        `helm install` of the pushed chart with zero further --set flags
        deploys exactly the image it was packaged with.

        Reuses the same image-publish path as `publish` (via the shared
        _publish_image helper — no duplicated build/tag/publish logic) and
        awaits it to completion before packaging the chart, so "image
        pushed before chart push" holds by construction.

        Args:
            helm_username: OCI Helm registry username.
            helm_password: OCI Helm registry password.
            registry: Image registry:repo the chart gets pinned to (same
                target `publish` would use).
            helm_registry: OCI Helm registry base, e.g. quay.io/weka.io/helm
                (matches kvlab's retired chart-push flow's default).
            replay_image: The replay image underneath, as for `publish`.
            version: Explicit version stamp for BOTH the image tag and the
                chart version/appVersion (e.g. a semver tag from the release
                workflow). Empty = content-hash scheme (v999.0.0-<sha12>).
            binaries: Prebuilt binaries, laid out as
                <name>_<os>_<arch>/kvlab (the release passes the ones it built
                for the GitHub Release). Omitted = compile from the Dockerfile.

        The image push uses the Helm credentials too: both registries are
        the same quay account.
        """
        image_name, version = await _publish_image(
            source, registry, replay_image, version, binaries,
            helm_username, helm_password,
        )

        chart_ref = await _package_and_push_chart(
            source.directory(CHART_DIR), CHART_NAME, registry, version,
            helm_registry, helm_username, helm_password,
        )

        return (
            f"Published kvlab image: {image_name}\n"
            f"Published Helm chart: {chart_ref} (pinned to image {image_name})"
        )

    @function
    async def build_router(
        self,
        source: Annotated[dagger.Directory, Ignore(SOURCE_IGNORE)],
        registry: str = "quay.io/weka.io/kvlab-router",
        version: str = "",
    ) -> dagger.Directory:
        """Builds the router image and packages chart/router WITHOUT pushing,
        returning a directory holding the packaged .tgz.

        The point is that it runs the same code as a release. Testing a chart by
        hand means reimplementing the version stamping and imageRepository sync
        that _package_chart does, and a local check that reimplements the
        pinning contract will agree with the released artifact right up until it
        matters. Needs no registry credentials.

            dagger call build-router --source=. export --path=./dist

        Args:
            registry: Image repo the chart is pinned to, as a release would.
            version: Explicit stamp; empty = content-hash scheme.
        """
        if not version:
            version = await _calc_version(source)
        # Built, not published: this proves the Dockerfile still builds, which
        # is the other half of what a pre-merge check is for. Both platforms,
        # because the publish ships both and a Go cross-compile can fail on one
        # target while the other is clean — checking only amd64 would let an
        # arm64 break through to the release.
        for p in PUBLISH_PLATFORMS:
            await (await _image(source, ROUTER_DOCKERFILE, p)).sync()
        packaged = _package_chart(source.directory(ROUTER_CHART_DIR), registry, version)
        return packaged.directory("/out")

    @function
    async def push_router_helm(
        self,
        source: Annotated[dagger.Directory, Ignore(SOURCE_IGNORE)],
        helm_username: dagger.Secret,
        helm_password: dagger.Secret,
        registry: str = "quay.io/weka.io/kvlab-router",
        helm_registry: str = "quay.io/weka.io/helm",
        version: str = "",
        binaries: dagger.Directory | None = None,
    ) -> str:
        """Publishes the REPLAY-LESS router image, then packages and pushes
        chart/router pinned to it.

        The mirror of `push_helm` for the router half of a release. Both run
        from the same commit under the same semver, so a release yields two
        images and two charts that are known to agree: the benchmark pair
        carries the embedded replay artifact, the router pair does not.

        Args:
            helm_username: OCI Helm registry username.
            helm_password: OCI Helm registry password.
            registry: Image registry:repo for the router image.
            helm_registry: OCI Helm registry base.
            version: Explicit version stamp for both image tag and chart
                version/appVersion. Empty = content-hash scheme.
            binaries: Prebuilt binaries, laid out as
                <name>_<os>_<arch>/kvlab (the release passes the ones it built
                for the GitHub Release). Omitted = compile from the Dockerfile.
        """
        image_name, version = await _publish_router_image(source, registry, version, binaries)

        chart_ref = await _package_and_push_chart(
            source.directory(ROUTER_CHART_DIR), ROUTER_CHART_NAME, registry, version,
            helm_registry, helm_username, helm_password,
        )

        return (
            f"Published router image: {image_name}\n"
            f"Published Helm chart: {chart_ref} (pinned to image {image_name})"
        )

    @function
    async def release(
        self,
        source: Annotated[dagger.Directory, Ignore(SOURCE_IGNORE)],
        helm_username: dagger.Secret,
        helm_password: dagger.Secret,
        version: str,
        registry: str = "quay.io/weka.io/kvlab",
        router_registry: str = "quay.io/weka.io/kvlab-router",
        helm_registry: str = "quay.io/weka.io/helm",
        replay_image: str = "quay.io/weka.io/kvlab:replay-40d-oct14-337a03810637",
        binaries: dagger.Directory | None = None,
    ) -> str:
        """The whole release: `push_helm` and `push_router_helm` in ONE engine
        session, the two pairs concurrently.

        One call rather than two so the engine starts and loads this module
        once, and the two pairs' builds and pushes overlap. Each
        pair keeps its own order — its image is published before its chart is
        pushed — and either pair failing fails the release, so it still ships a
        matched set or nothing.

        Args:
            version: The semver both images and both charts are published under.
            binaries: Prebuilt binaries, laid out as
                <name>_<os>_<arch>/kvlab (the release passes the ones it built
                for the GitHub Release). Omitted = compile from the Dockerfile.
        """
        bench, router = await asyncio.gather(
            self.push_helm(
                source, helm_username, helm_password,
                registry=registry, helm_registry=helm_registry,
                replay_image=replay_image, version=version, binaries=binaries,
            ),
            self.push_router_helm(
                source, helm_username, helm_password,
                registry=router_registry, helm_registry=helm_registry,
                version=version, binaries=binaries,
            ),
        )
        return f"{bench}\n{router}"
