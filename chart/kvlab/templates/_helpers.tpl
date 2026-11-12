{{/*
Expand the name of the chart.
*/}}
{{- define "kvlab.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "kvlab.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "kvlab.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "kvlab.labels" -}}
helm.sh/chart: {{ include "kvlab.chart" . }}
{{ include "kvlab.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "kvlab.selectorLabels" -}}
app.kubernetes.io/name: {{ include "kvlab.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Results are always written to resultsMountPath; the only question these two
helpers answer is what backs that path.

kvlab.results.ownClaim — non-empty ("true") when the chart should create and
own the results PVC. An existing resultsClaim always wins: the user manages
that volume's lifecycle, so provisioning a second one alongside it would be
silent waste. storeResults is the pre-existingClaim name for
createResultsClaim and is still honored, so values files written against
older chart versions keep their PVC instead of silently degrading to an
emptyDir that vanishes with the pod.
*/}}
{{- define "kvlab.results.ownClaim" -}}
{{- if and (not .Values.resultsClaim) (or .Values.createResultsClaim .Values.storeResults) -}}
true
{{- end -}}
{{- end }}

{{/*
kvlab.results.dataPath — where --save-request-data points, i.e.
resultsMountPath joined with resultsSubPath. kvlab creates its own
<run-timestamp>/ directory beneath this.

The subdirectory exists because the same PVC is normally shared across
purposes, and writing run directories straight at the volume root mixes them
in with everything else already there. The whole volume is still mounted at
resultsMountPath (deliberately NOT a volumeMount subPath), so the rest of the
PVC stays visible for kubectl exec/cp — only what kvlab WRITES is namespaced.

resultsSubPath is overridable. An EXPLICIT empty value writes straight to the
volume root (the pre-existing flat layout); a value that is absent or null
falls back to the default rather than the root, so a values file that never
mentions the key still gets the namespaced layout. Helm's `default` collapses
nil and "" into one case, which is exactly the distinction that matters here —
hence the kindIs "invalid" test.
*/}}
{{- define "kvlab.results.subPath" -}}
{{- if kindIs "invalid" .Values.resultsSubPath -}}
kvlab-requests-data
{{- else -}}
{{- trimAll "/" (toString .Values.resultsSubPath) -}}
{{- end -}}
{{- end }}

{{- define "kvlab.results.dataPath" -}}
{{- $sub := include "kvlab.results.subPath" . -}}
{{- if $sub -}}
{{- printf "%s/%s" (trimSuffix "/" .Values.resultsMountPath) $sub -}}
{{- else -}}
{{- .Values.resultsMountPath -}}
{{- end -}}
{{- end }}

{{/*
kvlab.results.claimName — the PVC to mount at resultsMountPath. Empty means
no PVC at all, i.e. an ephemeral emptyDir.
*/}}
{{- define "kvlab.results.claimName" -}}
{{- if .Values.resultsClaim -}}
{{- .Values.resultsClaim -}}
{{- else if include "kvlab.results.ownClaim" . -}}
{{- printf "%s-results" (include "kvlab.fullname" .) -}}
{{- end -}}
{{- end }}
