package benchmark

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/weka/kvlab/config"
)

// A replay request whose prompt is over the context limit retires the rest of
// its instance: prompts only grow within a session, so every later turn would
// overflow too. Such a request is neither completed nor an error, so it is
// counted here, written to the request data as its own row type, and reported
// on the progress line, in the summary and in the report: a limit below the
// corpus can otherwise drop most of a run without a trace.

const (
	recordTypeOverflow = "context_overflow"
	// ErrorKindContextOverflow is the error_kind of an overflow row.
	ErrorKindContextOverflow = "context_overflow"

	overflowReasonServer = "server_400"    // the server's context-length 400
	overflowReasonLimit  = "limit_context" // skipped by --limit-context before sending
)

// overflowRecord is one retirement, a request-data row of its own
// (record_type context_overflow) so readers counting requests never see it.
type overflowRecord struct {
	RecordType   string    `json:"record_type"`
	Time         time.Time `json:"time"`
	Model        string    `json:"model"`
	SeriesNum    int       `json:"series_num"`
	SeriesGUID   string    `json:"series_guid"`
	Turn         int       `json:"turn"`
	ErrorKind    string    `json:"error_kind"`
	Reason       string    `json:"reason"`
	ErrorMessage string    `json:"error_message,omitempty"`
	// PromptTokens is the request's prompt size in the capture.
	PromptTokens int `json:"prompt_tokens"`
	// MaxContext is the limit it went over: the server's (parsed from its 400)
	// or --limit-context; 0 when unknown.
	MaxContext int `json:"max_context,omitempty"`
	// RequestsDropped is this request plus the rest of its instance.
	RequestsDropped int `json:"requests_dropped"`
}

// overflowCounts is the run's tally.
type overflowCounts struct {
	sessions, requests, server, limit atomic.Int64
	maxContext                        atomic.Int64
}

func (c *overflowCounts) add(r overflowRecord) {
	c.sessions.Add(1)
	c.requests.Add(int64(r.RequestsDropped))
	if r.Reason == overflowReasonLimit {
		c.limit.Add(1)
	} else {
		c.server.Add(1)
	}
	if r.MaxContext > 0 {
		c.maxContext.Store(int64(r.MaxContext))
	}
}

// overflowSummary is the tally as the summary and the report show it.
type overflowSummary struct {
	Sessions     int64 `json:"sessions"`
	Requests     int64 `json:"requests"`
	Server       int64 `json:"server400"`
	LimitContext int64 `json:"limitContext"`
	MaxContext   int64 `json:"maxContext,omitempty"`
	// LongestPrompt is the largest prompt among the dropped requests.
	LongestPrompt int64 `json:"longestPrompt,omitempty"`
	// Warning is set when a round limit cut requests the corpus holds.
	Warning string `json:"warning,omitempty"`
}

func (c *overflowCounts) summary() overflowSummary {
	s := overflowSummary{Sessions: c.sessions.Load(), Requests: c.requests.Load(),
		Server: c.server.Load(), LimitContext: c.limit.Load(), MaxContext: c.maxContext.Load()}
	s.Warning = s.warning()
	return s
}

func (s *overflowSummary) addRecord(r overflowRecord) {
	s.Sessions++
	s.Requests += int64(r.RequestsDropped)
	if r.Reason == overflowReasonLimit {
		s.LimitContext++
	} else {
		s.Server++
	}
	if r.MaxContext > 0 {
		s.MaxContext = int64(r.MaxContext)
	}
	s.LongestPrompt = max(s.LongestPrompt, int64(r.PromptTokens))
	s.Warning = s.warning()
}

// warning names a context limit that cut sessions short.
func (s overflowSummary) warning() string {
	if s.Sessions == 0 {
		return ""
	}
	limit := "the context limit"
	if s.MaxContext > 0 {
		limit = fmt.Sprintf("max context %d", s.MaxContext)
		if isRoundLimit(s.MaxContext) {
			limit += " (a round server limit, likely --max-model-len)"
		}
	}
	return fmt.Sprintf("%d session(s) retired and %d request(s) never sent: the corpus has longer prompts than %s",
		s.Sessions, s.Requests, limit)
}

// isRoundLimit is a limit set by hand: a multiple of 1024.
func isRoundLimit(n int64) bool { return n > 0 && n%1024 == 0 }

// String is the summary line, "" with nothing retired.
func (s overflowSummary) String() string {
	if s.Sessions == 0 {
		return ""
	}
	out := fmt.Sprintf("%d session(s) retired (server 400: %d, --limit-context: %d), %d request(s) never sent",
		s.Sessions, s.Server, s.LimitContext, s.Requests)
	if s.MaxContext > 0 {
		out += fmt.Sprintf(", max context %d", s.MaxContext)
	}
	return out
}

var maxContextRe = regexp.MustCompile(`maximum context length is (\d+)`)

// maxContextOf is the limit a context-length 400 names, 0 if none.
func maxContextOf(err error) int {
	if err == nil {
		return 0
	}
	m := maxContextRe.FindStringSubmatch(err.Error())
	if m == nil {
		return 0
	}
	n, _ := strconv.Atoi(m[1])
	return n
}

// recordOverflow counts and writes one retirement: req is the request that
// went over, remaining the requests its instance would still have sent.
func recordOverflow(cfg AutoBenchmarkConfig, st *autoState, rdw *requestDataWriter, metrics RequestMetrics,
	req RouterReplayRequest, seriesNum int, seriesGUID string, turn, remaining int) {
	r := overflowRecord{
		RecordType:      recordTypeOverflow,
		Time:            time.Now(),
		Model:           cfg.Model,
		SeriesNum:       seriesNum,
		SeriesGUID:      seriesGUID,
		Turn:            turn,
		ErrorKind:       ErrorKindContextOverflow,
		Reason:          overflowReasonServer,
		PromptTokens:    req.InputTokens + req.CacheReadTokens + req.CacheCreationTokens,
		RequestsDropped: 1 + remaining,
	}
	if metrics.Skipped {
		r.Reason, r.MaxContext = overflowReasonLimit, cfg.LimitContext
	} else {
		r.ErrorMessage = errString(metrics.Error)
		r.MaxContext = maxContextOf(metrics.Error)
	}
	st.overflow.add(r)
	if rdw != nil {
		if err := rdw.writeAny(r); err != nil {
			fmt.Fprintf(os.Stderr, "warning: failed to write overflow record: %v\n", err)
		}
	}
}

// ---------------------------------------------------------------- start-up check

// fetchMaxModelLen reads max_model_len from an OpenAI-style models listing at
// base (vLLM reports it); 0 when the server does not say.
func fetchMaxModelLen(base string) (int, error) {
	base = strings.TrimRight(base, "/")
	urls := []string{base + "/models", base + "/v1/models"}
	if strings.HasSuffix(base, "/v1") {
		urls = urls[:1]
	}
	var lastErr error
	for _, u := range urls {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
		if err != nil {
			cancel()
			return 0, err
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			cancel()
			lastErr = err
			continue
		}
		var body struct {
			Data []struct {
				MaxModelLen int `json:"max_model_len"`
			} `json:"data"`
		}
		ok := resp.StatusCode/100 == 2
		if ok {
			err = json.NewDecoder(resp.Body).Decode(&body)
		}
		resp.Body.Close()
		cancel()
		if !ok {
			lastErr = fmt.Errorf("status %d from %s", resp.StatusCode, u)
			continue
		}
		if err != nil {
			return 0, err
		}
		if len(body.Data) == 0 {
			return 0, nil
		}
		return body.Data[0].MaxModelLen, nil
	}
	return 0, lastErr
}

// corpusFit is how much of a replay corpus a context limit admits.
type corpusFit struct {
	Requests, Sessions         int
	RequestsOver, SessionsOver int
	RequestsDropped            int // over the limit, plus the rest of their instance
	P50, P90, Max              int // prompt tokens per request
}

// scanCorpusFit reads every session of a replay file and measures its prompt
// sizes against limit.
func scanCorpusFit(path string, limit int) (corpusFit, error) {
	f, err := os.Open(path)
	if err != nil {
		return corpusFit{}, err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 1<<20)
	var fit corpusFit
	var sizes []int
	first := true
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 && !first {
			var sess struct {
				Instances []struct {
					Requests []struct {
						InputTokens         int `json:"input_tokens"`
						CacheReadTokens     int `json:"cache_read_tokens"`
						CacheCreationTokens int `json:"cache_creation_tokens"`
					} `json:"requests"`
				} `json:"instances"`
			}
			if json.Unmarshal(line, &sess) == nil {
				fit.Sessions++
				over := false
				for _, inst := range sess.Instances {
					for _, r := range inst.Requests {
						n := r.InputTokens + r.CacheReadTokens + r.CacheCreationTokens
						sizes = append(sizes, n)
						if n > limit {
							fit.RequestsOver++
							over = true
						}
					}
					// The first request over retires the rest of its instance.
					for i, r := range inst.Requests {
						if r.InputTokens+r.CacheReadTokens+r.CacheCreationTokens > limit {
							fit.RequestsDropped += len(inst.Requests) - i
							break
						}
					}
				}
				if over {
					fit.SessionsOver++
				}
			}
		}
		first = false
		if err != nil {
			break
		}
	}
	fit.Requests = len(sizes)
	if len(sizes) > 0 {
		sort.Ints(sizes)
		fit.P50 = sizes[len(sizes)/2]
		fit.P90 = sizes[len(sizes)*9/10]
		fit.Max = sizes[len(sizes)-1]
	}
	return fit, nil
}

// warning is the start-up warning, "" when the limit admits the corpus.
func (c corpusFit) warning(limit int) string {
	if c.RequestsOver == 0 {
		return ""
	}
	loud := ""
	if c.P50 > limit || c.P90 > limit {
		loud = "WARNING: "
	}
	return fmt.Sprintf("%smax_model_len %d is below %d of %d corpus requests (%.0f%%; prompt p50 %d, p90 %d, max %d tokens): "+
		"%d of %d sessions (%.0f%%) will be cut short and ~%d requests (%.0f%%) never sent",
		loud, limit, c.RequestsOver, c.Requests, pct(c.RequestsOver, c.Requests), c.P50, c.P90, c.Max,
		c.SessionsOver, c.Sessions, pct(c.SessionsOver, c.Sessions),
		c.RequestsDropped, pct(c.RequestsDropped, c.Requests))
}

func pct(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}

// checkContextFit warns at start-up when the server's max_model_len is below
// prompts the replay corpus holds: those sessions will be cut short.
func checkContextFit(cfg AutoBenchmarkConfig) {
	p, err := newReplayPoster(cfg.Model, config.GetAPIKeys(), "", "", false, 0, 0, 0, nil, nil)
	if err != nil {
		return
	}
	limit, err := fetchMaxModelLen(p.base)
	if err != nil || limit <= 0 {
		return
	}
	if cfg.LimitContext > 0 {
		limit = min(limit, cfg.LimitContext)
	}
	fit, err := scanCorpusFit(cfg.RouterReplayFile, limit)
	if err != nil {
		return
	}
	if w := fit.warning(limit); w != "" {
		fmt.Fprintf(os.Stderr, "[auto][%s] %s\n", shortModelName(cfg.Model), w)
	}
}
