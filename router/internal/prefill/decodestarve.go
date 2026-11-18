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
//
// The same samples give its mirror, underutilization: a node is idle at a
// sample with KV usage below IdleKV and nothing waiting, and the decode side
// is underutilized while the mean idle share over UnderutilWindowTime is at
// least IdleFraction (0, or above 1, turns it off). Then the window in force is
// UnderutilWindowTime, prefill nodes are never given back, and one is added
// only when prefill is slow (UnderutilTTFTHigh).

// DecodeProbe reads a decode node's KV cache usage (0 to 1) and the requests
// waiting on it.
type DecodeProbe func(ctx context.Context, b *registry.Backend) (kvUsage, waiting float64, err error)

type starveSample struct {
	at         time.Time
	frac, idle float64 // starved and idle shares of the decode nodes
}

// decodeSignals is the decode side over one window: the mean starved and idle
// shares, each with its flag, the threshold applied to that same mean.
type decodeSignals struct {
	starved, under        bool
	starvedFrac, idleFrac float64
	n                     int
}

// defaultStarveSpan is the starvation window under a count TTFT window.
const defaultStarveSpan = 30 * time.Second

func (m *Manager) starveSpan() time.Duration {
	if m.cfg.TTFTWindowTime > 0 {
		return m.cfg.TTFTWindowTime
	}
	return defaultStarveSpan
}

// underSpan is the window underutilization is judged on.
func (m *Manager) underSpan() time.Duration {
	if m.cfg.UnderutilWindowTime > 0 {
		return m.cfg.UnderutilWindowTime
	}
	return m.starveSpan()
}

// storeSpan is how long samples are kept: the longest window read.
func (m *Manager) storeSpan() time.Duration { return max(m.starveSpan(), m.underSpan()) }

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
	var starved, idle atomic.Int64
	nodeStarved := make([]bool, len(decode))
	var wg sync.WaitGroup
	for i, b := range decode {
		wg.Add(1)
		go func() {
			defer wg.Done()
			kv, waiting, err := probe(ctx, b)
			switch {
			case err != nil:
			case kv >= m.cfg.StarveKV && waiting >= m.cfg.StarveWaiting:
				starved.Add(1)
				nodeStarved[i] = true
			case kv < m.cfg.IdleKV && waiting == 0:
				idle.Add(1)
			}
		}()
	}
	wg.Wait()
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.cfg.Clock.Now()
	for i, b := range decode {
		m.nodeLocked(b).starved = nodeStarved[i]
	}
	n := float64(len(decode))
	m.starve = append(m.starve, starveSample{at: now,
		frac: float64(starved.Load()) / n, idle: float64(idle.Load()) / n})
	m.pruneStarveLocked(now)
}

func (m *Manager) pruneStarveLocked(now time.Time) {
	i := 0
	for i < len(m.starve) && now.Sub(m.starve[i].at) > m.storeSpan() {
		i++
	}
	if i > 0 {
		m.starve = append(m.starve[:0], m.starve[i:]...)
	}
}

// decodeSignalsLocked is the decode side over the samples of the last span
// (all taken since the last action, which restarts the window). No sample is
// neither starved nor underutilized.
func (m *Manager) decodeSignalsLocked(now time.Time, span time.Duration) decodeSignals {
	m.pruneStarveLocked(now)
	var s decodeSignals
	for _, x := range m.starve {
		if now.Sub(x.at) <= span {
			s.starvedFrac += x.frac
			s.idleFrac += x.idle
			s.n++
		}
	}
	if s.n == 0 {
		return s
	}
	s.starvedFrac /= float64(s.n)
	s.idleFrac /= float64(s.n)
	s.starved = s.starvedFrac >= m.cfg.StarveFraction
	s.under = m.cfg.IdleFraction > 0 && m.cfg.IdleFraction <= 1 && s.idleFrac >= m.cfg.IdleFraction
	return s
}

// decodeStarvedLocked is the starvation signal over the normal window.
func (m *Manager) decodeStarvedLocked(now time.Time) (bool, float64, int) {
	s := m.decodeSignalsLocked(now, m.starveSpan())
	return s.starved, s.starvedFrac, s.n
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
