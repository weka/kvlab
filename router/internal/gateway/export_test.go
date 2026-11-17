package gateway

import "net/http"

// recoverForTest exposes the recover boundary so the external test package can
// assert that a genuine panic still becomes a 500 while a client disconnect does
// not count as one.
func (s *Server) RecoverForTest(next http.Handler) http.Handler { return s.recoverMiddleware(next) }

// PDLegCharge exposes a dynamic P/D leg's budget charge.
func PDLegCharge(total, local, stored int64) int64 { return pdLegCharge(total, local, stored) }

// PDDiagConversation exposes the sampling of the per-leg charge log.
func PDDiagConversation(id string) (string, bool) { return pdDiagConversation(id) }
