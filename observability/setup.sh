#!/bin/bash

# T.A.R.S. Observability Stack Setup Script
# Deploys Grafana, Loki, Jaeger, and Prometheus alerting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-tars}"
HELM_TIMEOUT="${HELM_TIMEOUT:-10m}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}T.A.R.S. Observability Stack Setup${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for pods to be ready
wait_for_pods() {
    local app=$1
    local timeout=${2:-300}

    echo -e "${YELLOW}Waiting for $app pods to be ready...${NC}"
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=$app \
        -n $NAMESPACE \
        --timeout=${timeout}s || true
}

# Verify prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command_exists kubectl; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    exit 1
fi

if ! command_exists helm; then
    echo -e "${RED}Error: helm is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"
echo ""

# Create namespace if it doesn't exist
echo -e "${YELLOW}Creating namespace: $NAMESPACE${NC}"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓ Namespace ready${NC}"
echo ""

# Add Helm repositories
echo -e "${YELLOW}Adding Helm repositories...${NC}"
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo update
echo -e "${GREEN}✓ Helm repositories added${NC}"
echo ""

# ==========================================
# Deploy Prometheus (if not already present)
# ==========================================
echo -e "${YELLOW}Checking Prometheus installation...${NC}"
if helm list -n $NAMESPACE | grep -q prometheus; then
    echo -e "${GREEN}✓ Prometheus already installed${NC}"
else
    echo -e "${YELLOW}Installing Prometheus...${NC}"
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace $NAMESPACE \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.retention=14d \
        --set prometheus.prometheusSpec.retentionSize=50GB \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.accessModes[0]=ReadWriteOnce \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
        --timeout $HELM_TIMEOUT \
        --wait

    wait_for_pods prometheus 600
    echo -e "${GREEN}✓ Prometheus installed${NC}"
fi
echo ""

# ==========================================
# Apply Prometheus Alert Rules
# ==========================================
echo -e "${YELLOW}Applying Prometheus alert rules...${NC}"
kubectl apply -f "$SCRIPT_DIR/prom-alerts.yaml" -n $NAMESPACE
echo -e "${GREEN}✓ Alert rules configured${NC}"
echo ""

# ==========================================
# Deploy Loki
# ==========================================
echo -e "${YELLOW}Installing Loki...${NC}"
helm upgrade --install loki grafana/loki-stack \
    --namespace $NAMESPACE \
    --values "$SCRIPT_DIR/loki-values.yaml" \
    --timeout $HELM_TIMEOUT \
    --wait

wait_for_pods loki 300
echo -e "${GREEN}✓ Loki installed${NC}"
echo ""

# ==========================================
# Deploy Jaeger
# ==========================================
echo -e "${YELLOW}Installing Jaeger...${NC}"

# For development, use all-in-one. For production, use the production setup
if [ "${ENVIRONMENT:-dev}" = "production" ]; then
    # Install Elasticsearch for Jaeger storage (production)
    echo -e "${YELLOW}Installing Elasticsearch for Jaeger...${NC}"
    helm upgrade --install elasticsearch elastic/elasticsearch \
        --namespace $NAMESPACE \
        --set replicas=3 \
        --set minimumMasterNodes=2 \
        --set esJavaOpts="-Xmx1g -Xms1g" \
        --set resources.requests.memory=2Gi \
        --set resources.limits.memory=2Gi \
        --set volumeClaimTemplate.resources.requests.storage=100Gi \
        --timeout $HELM_TIMEOUT \
        --wait || echo -e "${YELLOW}Warning: Elasticsearch installation had issues${NC}"
fi

# Install Jaeger
helm upgrade --install jaeger jaegertracing/jaeger \
    --namespace $NAMESPACE \
    --values "$SCRIPT_DIR/jaeger-values.yaml" \
    --timeout $HELM_TIMEOUT \
    --wait

wait_for_pods jaeger 300
echo -e "${GREEN}✓ Jaeger installed${NC}"
echo ""

# ==========================================
# Deploy OpenTelemetry Collector
# ==========================================
echo -e "${YELLOW}Installing OpenTelemetry Collector...${NC}"
helm upgrade --install otel-collector open-telemetry/opentelemetry-collector \
    --namespace $NAMESPACE \
    --set mode=deployment \
    --set image.repository=otel/opentelemetry-collector-contrib \
    --set ports.otlp.enabled=true \
    --set ports.otlp.containerPort=4317 \
    --set ports.otlp.servicePort=4317 \
    --set ports.otlp-http.enabled=true \
    --set ports.otlp-http.containerPort=4318 \
    --set ports.otlp-http.servicePort=4318 \
    --set ports.jaeger-compact.enabled=true \
    --set ports.jaeger-compact.containerPort=6831 \
    --set ports.jaeger-compact.protocol=UDP \
    --set ports.jaeger-thrift.enabled=true \
    --set ports.jaeger-thrift.containerPort=14268 \
    --set ports.zipkin.enabled=true \
    --set ports.zipkin.containerPort=9411 \
    --timeout $HELM_TIMEOUT \
    --wait

wait_for_pods otel-collector 300
echo -e "${GREEN}✓ OpenTelemetry Collector installed${NC}"
echo ""

# ==========================================
# Deploy Grafana
# ==========================================
echo -e "${YELLOW}Installing Grafana...${NC}"

# Check if Grafana is already installed via Prometheus stack
if kubectl get deployment -n $NAMESPACE | grep -q grafana; then
    echo -e "${GREEN}✓ Grafana already installed via Prometheus stack${NC}"
else
    helm upgrade --install grafana grafana/grafana \
        --namespace $NAMESPACE \
        --set persistence.enabled=true \
        --set persistence.size=10Gi \
        --set adminPassword=admin \
        --set service.type=ClusterIP \
        --set datasources."datasources\.yaml".apiVersion=1 \
        --set datasources."datasources\.yaml".datasources[0].name=Prometheus \
        --set datasources."datasources\.yaml".datasources[0].type=prometheus \
        --set datasources."datasources\.yaml".datasources[0].url=http://prometheus-kube-prometheus-prometheus:9090 \
        --set datasources."datasources\.yaml".datasources[0].isDefault=true \
        --set datasources."datasources\.yaml".datasources[1].name=Loki \
        --set datasources."datasources\.yaml".datasources[1].type=loki \
        --set datasources."datasources\.yaml".datasources[1].url=http://loki:3100 \
        --set datasources."datasources\.yaml".datasources[2].name=Jaeger \
        --set datasources."datasources\.yaml".datasources[2].type=jaeger \
        --set datasources."datasources\.yaml".datasources[2].url=http://jaeger-query:16686 \
        --timeout $HELM_TIMEOUT \
        --wait

    wait_for_pods grafana 300
fi
echo -e "${GREEN}✓ Grafana installed${NC}"
echo ""

# ==========================================
# Import Grafana Dashboards
# ==========================================
echo -e "${YELLOW}Importing Grafana dashboards...${NC}"

# Get Grafana pod name
GRAFANA_POD=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=grafana -o jsonpath='{.items[0].metadata.name}')

if [ -n "$GRAFANA_POD" ]; then
    # Copy dashboard to pod
    kubectl cp "$SCRIPT_DIR/grafana-dashboard.json" \
        $NAMESPACE/$GRAFANA_POD:/tmp/dashboard.json

    # Import dashboard via Grafana API
    kubectl exec -n $NAMESPACE $GRAFANA_POD -- \
        curl -X POST \
        -H "Content-Type: application/json" \
        -d @/tmp/dashboard.json \
        http://admin:admin@localhost:3000/api/dashboards/db \
        || echo -e "${YELLOW}Warning: Dashboard import may need manual completion${NC}"

    echo -e "${GREEN}✓ Dashboard imported${NC}"
else
    echo -e "${YELLOW}Warning: Could not find Grafana pod for dashboard import${NC}"
fi
echo ""

# ==========================================
# Create ServiceMonitors
# ==========================================
echo -e "${YELLOW}Creating ServiceMonitors for T.A.R.S. services...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: tars-backend
  namespace: $NAMESPACE
  labels:
    app: tars-backend
spec:
  selector:
    matchLabels:
      app: tars-backend
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: tars-ui
  namespace: $NAMESPACE
  labels:
    app: tars-ui
spec:
  selector:
    matchLabels:
      app: tars-ui
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF

echo -e "${GREEN}✓ ServiceMonitors created${NC}"
echo ""

# ==========================================
# Summary
# ==========================================
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Observability Stack Installation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "Components installed:"
echo -e "  ${GREEN}✓${NC} Prometheus (metrics collection)"
echo -e "  ${GREEN}✓${NC} Grafana (visualization)"
echo -e "  ${GREEN}✓${NC} Loki (log aggregation)"
echo -e "  ${GREEN}✓${NC} Jaeger (distributed tracing)"
echo -e "  ${GREEN}✓${NC} OpenTelemetry Collector (telemetry pipeline)"
echo -e "  ${GREEN}✓${NC} Alert Rules (configured)"
echo ""
echo -e "${YELLOW}Access Information:${NC}"
echo ""

# Get Grafana password
GRAFANA_PASSWORD=$(kubectl get secret --namespace $NAMESPACE grafana -o jsonpath="{.data.admin-password}" 2>/dev/null | base64 --decode || echo "admin")

echo -e "${GREEN}Grafana:${NC}"
echo -e "  Port Forward: kubectl port-forward -n $NAMESPACE svc/grafana 3000:80"
echo -e "  URL: http://localhost:3000"
echo -e "  Username: admin"
echo -e "  Password: $GRAFANA_PASSWORD"
echo ""

echo -e "${GREEN}Prometheus:${NC}"
echo -e "  Port Forward: kubectl port-forward -n $NAMESPACE svc/prometheus-kube-prometheus-prometheus 9090:9090"
echo -e "  URL: http://localhost:9090"
echo ""

echo -e "${GREEN}Jaeger UI:${NC}"
echo -e "  Port Forward: kubectl port-forward -n $NAMESPACE svc/jaeger-query 16686:16686"
echo -e "  URL: http://localhost:16686"
echo ""

echo -e "${GREEN}Loki:${NC}"
echo -e "  Port Forward: kubectl port-forward -n $NAMESPACE svc/loki 3100:3100"
echo -e "  URL: http://localhost:3100"
echo ""

echo -e "${YELLOW}Next Steps:${NC}"
echo -e "1. Update T.A.R.S. backend to export metrics to Prometheus"
echo -e "2. Configure OpenTelemetry instrumentation in application code"
echo -e "3. Access Grafana and verify dashboards are populated"
echo -e "4. Test alert firing by simulating high latency/errors"
echo -e "5. Review Loki logs and Jaeger traces"
echo ""

echo -e "${GREEN}Setup complete!${NC}"
