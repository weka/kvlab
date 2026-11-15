package gateway_test

import "testing"

// --prefill-split under the default cost routing. The prefill job's node and
// the decode resend are both picked by the pool's flow, which here is
// selectByCost, so the scenarios of the ladder must hold unchanged.
func TestPrefillSplitUnderCostRouting(t *testing.T) {
	for name, fn := range map[string]func(*testing.T){
		"heavy prefill runs on a drained node":          TestHeavyPrefillRunsOnADrainedNodeThenDecodesElsewhere,
		"decode avoids a prefill node":                  TestDecodeRoutingAvoidsAPrefillNodeWhileItHasJobs,
		"no prefill node falls back to the original":    TestNoPrefillNodeFallsBackToTheOriginalWithBypass,
		"rejection of a bypassed request does not loop": TestRejectionOfABypassedRequestDoesNotLoop,
		"jobs never exceed the node limit":              TestPrefillJobsNeverExceedTheNodeConcurrencyLimit,
		"429 job is retried":                            TestPrefillJobRefusedWith429IsRetriedNotFailed,
		"split off passes the rejection through":        TestSplitOffPassesTheRejectionThrough,
	} {
		t.Run(name, func(t *testing.T) {
			splitCostRouting = true
			t.Cleanup(func() { splitCostRouting = false })
			fn(t)
		})
	}
}
