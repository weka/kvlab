package prefill

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/weka/kvlab/router/internal/registry"
)

// The decode-starvation signal for dynamic P/D (Config.PD).
//
// Every DecodeSampleInterval each healthy decode node is probed once. A node
// is starved at a sample when its KV cache usage is at least StarveKV and it
// has at least StarveWaiting requests waiting: it can take no more decodes.
// The sample is the share of decode nodes starved. The signal is the mean of
// the samples in the provisioning window (TTFTWindowTime, or 30s with a count
// window), which restarts at every provisioning action like the TTFT window;
// the decode side is starved while that mean is at least StarveFraction. No
// sample is not starved. It changes the provisioning rules (pdttft.go): a
// starved decode side gets prefill nodes back unless prefill is very slow.

// DecodeProbe reads a decode node's KV cache usage (0 to 1) and the requests
// waiting on it.
type DecodeProbe func(ctx context.Context, b *registry.Backend) (kvUsage, waiting float64, err error)

type starveSample struct {
	at   time.Time
	frac float64
}

// defaultStarveSpan is the starvation window under a count TTFT window.
const defaultStarveSpan = 30 * time.Second

func (m *Manager) starveSpan() time.Duration {
	if m.cfg.TTFTWindowTime > 0 {
		return m.cfg.TTFTWindowTime
	}
	return defaultStarveSpan
}

// SampleDecode probes every decode node of cands once, concurrently, and adds
// the starved share to the window. A node that cannot be probed counts as not
// starved. With no decode node there is no sample.
func (m *Manager) SampleDecode(ctx context.Context, cands []*registry.Backend) {
	probe := m.cfg.DecodeProbe
	if probe == nil {
		return
	}
	m.mu.Lock()
	var decode []*registry.Backend
	for _, b := range cands {
		if m.nodeLocked(b).mode == Decode {
			decode = append(decode, b)
		}
	}
	m.mu.Unlock()
	if len(decode) == 0 {
		return
	}
	var starved atomic.Int64
	var wg sync.WaitGroup
	for _, b := range decode {
		wg.Add(1)
		go func() {
			defer wg.Done()
			kv, waiting, err := probe(ctx, b)
			if err == nil && kv >= m.cfg.StarveKV && waiting >= m.cfg.StarveWaiting {
				starved.Add(1)
			}
		}()
	}
	wg.Wait()
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.cfg.Clock.Now()
	m.starve = append(m.starve, starveSample{at: now, frac: float64(starved.Load()) / float64(len(decode))})
	m.pruneStarveLocked(now)
}

func (m *Manager) pruneStarveLocked(now time.Time) {
	i := 0
	for i < len(m.starve) && now.Sub(m.starve[i].at) > m.starveSpan() {
		i++
	}
	if i > 0 {
		m.starve = append(m.starve[:0], m.starve[i:]...)
	}
}

// decodeStarvedLocked is the starvation signal now: whether the decode side
// counts as starved, the window's mean starved share, and its sample count.
func (m *Manager) decodeStarvedLocked(now time.Time) (bool, float64, int) {
	m.pruneStarveLocked(now)
	if len(m.starve) == 0 {
		return false, 0, 0
	}
	var sum float64
	for _, s := range m.starve {
		sum += s.frac
	}
	mean := sum / float64(len(m.starve))
	return mean >= m.cfg.StarveFraction, mean, len(m.starve)
}

// runDecodeSampler samples every DecodeSampleInterval until ctx is done.
func (m *Manager) runDecodeSampler(ctx context.Context, known func() []*registry.Backend) {
	t := m.cfg.Clock.NewTicker(m.cfg.DecodeSampleInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C():
			sctx, cancel := context.WithTimeout(ctx, m.cfg.DecodeSampleInterval)
			m.SampleDecode(sctx, available(known()))
			cancel()
		}
	}
}

// HTTPDecodeProbe reads a vLLM node's /metrics: vllm:kv_cache_usage_perc
// (vllm:gpu_cache_usage_perc on older vLLM; a 0-1 fraction despite the
// name), the highest across its engines, and vllm:num_requests_waiting, summed.
func HTTPDecodeProbe(client *http.Client) DecodeProbe {
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Second}
	}
	return func(ctx context.Context, b *registry.Backend) (float64, float64, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, registry.ResolveURL(b.URL, "/metrics"), nil)
		if err != nil {
			return 0, 0, err
		}
		resp, err := client.Do(req)
		if err != nil {
			return 0, 0, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return 0, 0, fmt.Errorf("decode probe %s: status %d", b.URL, resp.StatusCode)
		}
		var kv, waiting float64
		var sawKV bool
		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 64<<10), 1<<20)
		for sc.Scan() {
			line := sc.Text()
			if strings.HasPrefix(line, "#") {
				continue
			}
			name, val, ok := metricLine(line)
			if !ok {
				continue
			}
			switch name {
			case "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc":
				kv, sawKV = max(kv, val), true
			case "vllm:num_requests_waiting":
				waiting += val
			}
		}
		if !sawKV {
			return 0, 0, fmt.Errorf("decode probe %s: no KV cache usage", b.URL)
		}
		return kv, waiting, nil
	}
}

func metricLine(line string) (string, float64, bool) {
	sp := strings.LastIndexByte(line, ' ')
	if sp < 0 {
		return "", 0, false
	}
	series := line[:sp]
	name := series
	if i := strings.IndexByte(series, '{'); i >= 0 {
		name = series[:i]
	}
	v, err := strconv.ParseFloat(line[sp+1:], 64)
	return name, v, err == nil
}
