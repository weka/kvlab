// Command graft publishes a multi-arch image whose bottom layers are an
// existing image's, referenced by digest, without ever downloading them.
//
// The kvlab benchmark image is a 1.3 GB (9.5 GB unpacked) replay layer plus a
// few tens of MB of runtime. The replay layer is already in the registry as the
// replay image. Building the image in an engine means pulling that layer into
// the engine first, minutes on a fresh CI runner, only to push it back to the
// registry that already has it. graft instead takes the runtime image (an OCI
// layout from the build), and for each platform in it writes a manifest that
// lists the base image's layers followed by the runtime's, with the runtime's
// config. The push uploads the runtime layers; for the base layers the registry
// is only asked whether it has them (or to mount them from another repo).
//
//	graft -layout DIR -base REF -dest REF [-insecure]
//
// Credentials come from GRAFT_USERNAME / GRAFT_PASSWORD when set (applied to
// every registry the command talks to), otherwise the default keychain.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/empty"
	"github.com/google/go-containerregistry/pkg/v1/layout"
	"github.com/google/go-containerregistry/pkg/v1/mutate"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

func main() {
	layoutDir := flag.String("layout", "", "OCI layout holding the runtime image index (one manifest per platform)")
	baseRef := flag.String("base", "", "image whose layers go at the bottom of every platform")
	destRef := flag.String("dest", "", "where to publish the resulting index")
	insecure := flag.Bool("insecure", false, "allow plain-HTTP registries")
	flag.Parse()
	if *layoutDir == "" || *baseRef == "" || *destRef == "" {
		flag.Usage()
		os.Exit(2)
	}
	digest, err := graft(*layoutDir, *baseRef, *destRef, *insecure)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s@%s\n", *destRef, digest)
}

func graft(layoutDir, baseRef, destRef string, insecure bool) (v1.Hash, error) {
	var nameOpts []name.Option
	if insecure {
		nameOpts = append(nameOpts, name.Insecure)
	}
	auth := remote.WithAuthFromKeychain(authn.DefaultKeychain)
	if u := os.Getenv("GRAFT_USERNAME"); u != "" {
		auth = remote.WithAuth(&authn.Basic{Username: u, Password: os.Getenv("GRAFT_PASSWORD")})
	}

	bref, err := name.ParseReference(baseRef, nameOpts...)
	if err != nil {
		return v1.Hash{}, err
	}
	dref, err := name.ParseReference(destRef, nameOpts...)
	if err != nil {
		return v1.Hash{}, err
	}
	// Only the base's manifest and config are fetched. Its layers are used by
	// descriptor; their blobs are never read.
	base, err := remote.Image(bref, auth)
	if err != nil {
		return v1.Hash{}, fmt.Errorf("base %s: %w", baseRef, err)
	}
	baseCfg, err := base.ConfigFile()
	if err != nil {
		return v1.Hash{}, err
	}

	path, err := layout.FromPath(layoutDir)
	if err != nil {
		return v1.Hash{}, err
	}
	src, err := path.ImageIndex()
	if err != nil {
		return v1.Hash{}, err
	}
	srcManifest, err := src.IndexManifest()
	if err != nil {
		return v1.Hash{}, err
	}
	// A layout's top-level index may wrap the real one.
	if len(srcManifest.Manifests) == 1 && srcManifest.Manifests[0].MediaType.IsIndex() {
		if src, err = src.ImageIndex(srcManifest.Manifests[0].Digest); err != nil {
			return v1.Hash{}, err
		}
		if srcManifest, err = src.IndexManifest(); err != nil {
			return v1.Hash{}, err
		}
	}

	out := mutate.IndexMediaType(empty.Index, types.OCIImageIndex)
	for _, desc := range srcManifest.Manifests {
		if !desc.MediaType.IsImage() {
			continue
		}
		app, err := src.Image(desc.Digest)
		if err != nil {
			return v1.Hash{}, err
		}
		img, err := onto(base, baseCfg, app)
		if err != nil {
			return v1.Hash{}, fmt.Errorf("platform %v: %w", desc.Platform, err)
		}
		cfg, err := img.ConfigFile()
		if err != nil {
			return v1.Hash{}, err
		}
		out = mutate.AppendManifests(out, mutate.IndexAddendum{
			Add: img,
			Descriptor: v1.Descriptor{Platform: &v1.Platform{
				OS: cfg.OS, Architecture: cfg.Architecture, Variant: cfg.Variant,
			}},
		})
	}
	if err := remote.WriteIndex(dref, out, auth); err != nil {
		return v1.Hash{}, fmt.Errorf("push %s: %w", destRef, err)
	}
	return out.Digest()
}

// onto returns app with base's layers underneath its own, and app's config
// (platform, entrypoint, env, labels, user, ...) with the layer list and
// history extended to match.
func onto(base v1.Image, baseCfg *v1.ConfigFile, app v1.Image) (v1.Image, error) {
	appCfg, err := app.ConfigFile()
	if err != nil {
		return nil, err
	}
	layers, err := app.Layers()
	if err != nil {
		return nil, err
	}
	img, err := mutate.AppendLayers(base, layers...)
	if err != nil {
		return nil, err
	}
	merged, err := img.ConfigFile()
	if err != nil {
		return nil, err
	}
	cfg := appCfg.DeepCopy()
	cfg.RootFS.DiffIDs = merged.RootFS.DiffIDs
	cfg.History = append(append([]v1.History{}, baseCfg.History...), appCfg.History...)
	img, err = mutate.ConfigFile(img, cfg)
	if err != nil {
		return nil, err
	}
	img = mutate.MediaType(img, types.OCIManifestSchema1)
	return mutate.ConfigMediaType(img, types.OCIConfigJSON), nil
}
