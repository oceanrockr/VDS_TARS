{{/*
Expand the name of the chart.
*/}}
{{- define "tars.name" -}}
{{- default .Chart.Name .Values.global.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "tars.fullname" -}}
{{- if .Values.global.fullnameOverride }}
{{- .Values.global.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.global.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "tars.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "tars.labels" -}}
helm.sh/chart: {{ include "tars.chart" . }}
{{ include "tars.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "tars.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tars.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Backend selector labels
*/}}
{{- define "tars.backend.selectorLabels" -}}
app: tars-backend
app.kubernetes.io/name: {{ include "tars.name" . }}
app.kubernetes.io/component: backend
{{- end }}

{{/*
PostgreSQL selector labels
*/}}
{{- define "tars.postgresql.selectorLabels" -}}
app: tars-postgres
app.kubernetes.io/name: {{ include "tars.name" . }}
app.kubernetes.io/component: database
{{- end }}

{{/*
Redis selector labels
*/}}
{{- define "tars.redis.selectorLabels" -}}
app: tars-redis
app.kubernetes.io/name: {{ include "tars.name" . }}
app.kubernetes.io/component: cache
{{- end }}

{{/*
ChromaDB selector labels
*/}}
{{- define "tars.chromadb.selectorLabels" -}}
app: tars-chromadb
app.kubernetes.io/name: {{ include "tars.name" . }}
app.kubernetes.io/component: vector-db
{{- end }}

{{/*
Ollama selector labels
*/}}
{{- define "tars.ollama.selectorLabels" -}}
app: tars-ollama
app.kubernetes.io/name: {{ include "tars.name" . }}
app.kubernetes.io/component: llm
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "tars.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "tars.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Backend service name
*/}}
{{- define "tars.backend.serviceName" -}}
{{- printf "%s-backend" (include "tars.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
PostgreSQL service name
*/}}
{{- define "tars.postgresql.serviceName" -}}
{{- printf "%s-postgres" (include "tars.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Redis service name
*/}}
{{- define "tars.redis.serviceName" -}}
{{- printf "%s-redis" (include "tars.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
ChromaDB service name
*/}}
{{- define "tars.chromadb.serviceName" -}}
{{- printf "%s-chromadb" (include "tars.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Ollama service name
*/}}
{{- define "tars.ollama.serviceName" -}}
{{- printf "%s-ollama" (include "tars.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Namespace
*/}}
{{- define "tars.namespace" -}}
{{- default .Release.Namespace .Values.global.namespace }}
{{- end }}
