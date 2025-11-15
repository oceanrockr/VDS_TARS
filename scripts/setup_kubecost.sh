#!/bin/bash

# Kubecost Setup Script for T.A.R.S.
# Provides comprehensive cost tracking and optimization recommendations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-kubecost}"
HELM_TIMEOUT="${HELM_TIMEOUT:-10m}"
KUBECOST_VERSION="${KUBECOST_VERSION:-2.1.0}"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Kubecost Setup for T.A.R.S.${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
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

# Create namespace
echo -e "${YELLOW}Creating namespace: $NAMESPACE${NC}"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓ Namespace ready${NC}"
echo ""

# Add Kubecost Helm repository
echo -e "${YELLOW}Adding Kubecost Helm repository...${NC}"
helm repo add kubecost https://kubecost.github.io/cost-analyzer/
helm repo update
echo -e "${GREEN}✓ Repository added${NC}"
echo ""

# Create Kubecost values file
echo -e "${YELLOW}Creating Kubecost configuration...${NC}"

cat > /tmp/kubecost-values.yaml <<'EOF'
# Kubecost Configuration for T.A.R.S.

global:
  # Grafana integration
  grafana:
    enabled: true
    domainName: grafana.tars.svc.cluster.local

  # Prometheus integration
  prometheus:
    enabled: true
    fqdn: http://prometheus-kube-prometheus-prometheus.tars.svc.cluster.local:9090

# Kubecost settings
kubecostModel:
  image: gcr.io/kubecost1/cost-model
  imagePullPolicy: Always

  # Resource allocation
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 2Gi

  # Cost configuration
  costConfig:
    # Cloud provider (on-prem)
    provider: "custom"

    # Custom pricing (adjust based on your infrastructure costs)
    CPU: "0.031611"          # $/core/hour (AWS m5.xlarge equivalent)
    RAM: "0.004237"          # $/GB/hour
    GPU: "0.95"              # $/GPU/hour (Tesla T4 equivalent)
    storage: "0.00014"       # $/GB/hour (EBS gp3 equivalent)
    network: "0.01"          # $/GB transferred

    # Discount rates
    spotCPU: "0.015806"      # 50% discount for spot instances
    spotRAM: "0.002119"

    # On-demand rates (if applicable)
    customPricesEnabled: true

  # Allocation settings
  allocation:
    # Include idle costs
    includeIdleCost: true

    # Allocation method
    allocateIdleByNode: true

# Cost analyzer frontend
costAnalyzerFrontend:
  image: gcr.io/kubecost1/frontend
  imagePullPolicy: Always

  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi

# Service configuration
service:
  type: ClusterIP
  port: 9090
  targetPort: 9090

  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"

# Ingress
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: kubecost.tars.local
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: kubecost-tls
      hosts:
        - kubecost.tars.local

# Persistence
persistentVolume:
  enabled: true
  size: 32Gi
  storageClass: ""
  accessModes:
    - ReadWriteOnce

# Metrics retention
prometheus:
  server:
    retention: "14d"
    persistentVolume:
      enabled: true
      size: 50Gi

# Reporting
reporting:
  # Daily reports
  daily:
    enabled: true

  # Weekly reports
  weekly:
    enabled: true

  # Monthly reports
  monthly:
    enabled: true

  # Email notifications (configure SMTP)
  email:
    enabled: false
    # smtpHost: smtp.gmail.com
    # smtpPort: 587
    # from: kubecost@example.com
    # to: ["admin@example.com"]

# Alerts
alerts:
  enabled: true

  # Budget alerts
  budget:
    - name: "tars-monthly-budget"
      threshold: 1000  # USD
      window: "month"

    - name: "tars-daily-budget"
      threshold: 50    # USD
      window: "day"

  # Efficiency alerts
  efficiency:
    - name: "low-cpu-efficiency"
      threshold: 0.3   # 30%

    - name: "low-memory-efficiency"
      threshold: 0.4   # 40%

# Network costs
networkCosts:
  enabled: true

  # Pod network costs
  podMonitor:
    enabled: true

# Cluster controller (for optimization recommendations)
clusterController:
  enabled: true

  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi

# ServiceMonitor for Prometheus
serviceMonitor:
  enabled: true

  additionalLabels:
    release: prometheus

# RBAC
rbac:
  create: true

# Service Account
serviceAccount:
  create: true
  name: kubecost

# Node selector (optional)
nodeSelector: {}

# Tolerations (optional)
tolerations: []

# Affinity (optional)
affinity: {}
EOF

echo -e "${GREEN}✓ Configuration created${NC}"
echo ""

# Install Kubecost
echo -e "${YELLOW}Installing Kubecost...${NC}"

helm upgrade --install kubecost kubecost/cost-analyzer \
    --namespace $NAMESPACE \
    --values /tmp/kubecost-values.yaml \
    --version $KUBECOST_VERSION \
    --timeout $HELM_TIMEOUT \
    --wait

echo -e "${GREEN}✓ Kubecost installed${NC}"
echo ""

# Wait for pods to be ready
echo -e "${YELLOW}Waiting for Kubecost pods to be ready...${NC}"
kubectl wait --for=condition=ready pod \
    -l app=cost-analyzer \
    -n $NAMESPACE \
    --timeout=300s || true

echo -e "${GREEN}✓ Kubecost is ready${NC}"
echo ""

# Create cost allocation labels
echo -e "${YELLOW}Setting up cost allocation labels...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: kubecost-allocation-labels
  namespace: $NAMESPACE
data:
  labels.yaml: |
    # T.A.R.S. Cost Allocation Labels

    # Application labels
    app: "Application name"
    component: "Component within application"
    tier: "Application tier (frontend, backend, database)"

    # Environment labels
    environment: "Environment (production, staging, dev)"
    region: "Geographic region (us-east, us-west)"

    # Team labels
    team: "Team responsible"
    owner: "Cost center owner"

    # Resource labels
    workload-type: "Workload type (deployment, statefulset, daemonset)"
    gpu-enabled: "GPU-enabled workload"
EOF

echo -e "${GREEN}✓ Cost allocation labels configured${NC}"
echo ""

# Create sample cost optimization report
echo -e "${YELLOW}Creating cost optimization report...${NC}"

cat > /tmp/kubecost-report.sh <<'SCRIPT'
#!/bin/bash

# Kubecost Cost Report Generator

API_ENDPOINT="http://kubecost-cost-analyzer.kubecost.svc.cluster.local:9090"

echo "=== T.A.R.S. Cost Report ==="
echo ""

# Total costs
echo "Total Cluster Costs (Last 7 days):"
curl -s "${API_ENDPOINT}/model/aggregatedCostModel?window=7d" | jq -r '.data[] | "\(.name): $\(.totalCost)"'
echo ""

# Namespace costs
echo "Namespace Costs (Last 7 days):"
curl -s "${API_ENDPOINT}/model/allocation?window=7d&aggregate=namespace" | jq -r '.data[] | "\(.name): $\(.totalCost)"'
echo ""

# Pod costs (top 10)
echo "Top 10 Most Expensive Pods (Last 7 days):"
curl -s "${API_ENDPOINT}/model/allocation?window=7d&aggregate=pod" | jq -r '.data | sort_by(-.totalCost) | .[:10][] | "\(.name): $\(.totalCost)"'
echo ""

# Efficiency metrics
echo "Resource Efficiency:"
curl -s "${API_ENDPOINT}/model/savings" | jq -r '.data | "CPU Efficiency: \(.cpuEfficiency*100)%\nMemory Efficiency: \(.ramEfficiency*100)%"'
echo ""

# Savings recommendations
echo "Cost Optimization Recommendations:"
curl -s "${API_ENDPOINT}/model/savings/requestSizing" | jq -r '.data[] | "- \(.message): Potential savings $\(.savings)"'
echo ""
SCRIPT

chmod +x /tmp/kubecost-report.sh
cp /tmp/kubecost-report.sh /usr/local/bin/kubecost-report || true

echo -e "${GREEN}✓ Report script created${NC}"
echo ""

# Summary
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Kubecost Installation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

echo -e "${YELLOW}Access Information:${NC}"
echo ""

echo -e "${GREEN}Kubecost UI:${NC}"
echo -e "  Port Forward: kubectl port-forward -n $NAMESPACE svc/kubecost-cost-analyzer 9090:9090"
echo -e "  URL: http://localhost:9090"
echo ""

echo -e "${YELLOW}Usage:${NC}"
echo -e "  • View cost allocation: http://localhost:9090/allocation.html"
echo -e "  • View savings recommendations: http://localhost:9090/savings.html"
echo -e "  • View cost reports: http://localhost:9090/reports.html"
echo -e "  • Generate CLI report: kubecost-report"
echo ""

echo -e "${YELLOW}API Examples:${NC}"
echo -e "  # Get total costs"
echo -e "  curl http://localhost:9090/model/allocation?window=7d"
echo ""
echo -e "  # Get namespace costs"
echo -e "  curl http://localhost:9090/model/allocation?window=7d&aggregate=namespace"
echo ""
echo -e "  # Get savings recommendations"
echo -e "  curl http://localhost:9090/model/savings"
echo ""

echo -e "${GREEN}Setup complete!${NC}"
