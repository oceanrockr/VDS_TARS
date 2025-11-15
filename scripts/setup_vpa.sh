#!/bin/bash

# Vertical Pod Autoscaler (VPA) Setup Script for T.A.R.S.
# Automatically right-sizes pod resource requests and limits

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-tars}"
VPA_VERSION="${VPA_VERSION:-1.0.0}"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Vertical Pod Autoscaler Setup for T.A.R.S.${NC}"
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

if ! command_exists git; then
    echo -e "${RED}Error: git is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"
echo ""

# Clone VPA repository
echo -e "${YELLOW}Cloning VPA repository...${NC}"

VPA_DIR="/tmp/autoscaler"
if [ -d "$VPA_DIR" ]; then
    rm -rf "$VPA_DIR"
fi

git clone https://github.com/kubernetes/autoscaler.git "$VPA_DIR"
cd "$VPA_DIR/vertical-pod-autoscaler"

echo -e "${GREEN}✓ Repository cloned${NC}"
echo ""

# Install VPA
echo -e "${YELLOW}Installing VPA components...${NC}"

./hack/vpa-up.sh

echo -e "${GREEN}✓ VPA installed${NC}"
echo ""

# Wait for VPA pods to be ready
echo -e "${YELLOW}Waiting for VPA pods to be ready...${NC}"

kubectl wait --for=condition=ready pod \
    -l app=vpa-admission-controller \
    -n kube-system \
    --timeout=300s || true

kubectl wait --for=condition=ready pod \
    -l app=vpa-recommender \
    -n kube-system \
    --timeout=300s || true

kubectl wait --for=condition=ready pod \
    -l app=vpa-updater \
    -n kube-system \
    --timeout=300s || true

echo -e "${GREEN}✓ VPA components are ready${NC}"
echo ""

# Create VPA configurations for T.A.R.S. components
echo -e "${YELLOW}Creating VPA configurations for T.A.R.S. components...${NC}"

# Backend VPA
cat <<EOF | kubectl apply -f -
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: tars-backend-vpa
  namespace: $NAMESPACE
  labels:
    app: tars-backend
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tars-backend

  updatePolicy:
    updateMode: "Auto"  # Auto, Recreate, Initial, or Off

  resourcePolicy:
    containerPolicies:
      - containerName: backend
        minAllowed:
          cpu: 500m
          memory: 512Mi
        maxAllowed:
          cpu: 4000m
          memory: 8Gi
        controlledResources:
          - cpu
          - memory
        mode: Auto

  # Recommendation mode
  recommenders:
    - name: default
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: tars-ui-vpa
  namespace: $NAMESPACE
  labels:
    app: tars-ui
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tars-ui

  updatePolicy:
    updateMode: "Auto"

  resourcePolicy:
    containerPolicies:
      - containerName: ui
        minAllowed:
          cpu: 100m
          memory: 128Mi
        maxAllowed:
          cpu: 1000m
          memory: 2Gi
        controlledResources:
          - cpu
          - memory
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: postgres-vpa
  namespace: $NAMESPACE
  labels:
    app: postgresql
spec:
  targetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: postgres-primary

  updatePolicy:
    updateMode: "Initial"  # Don't auto-update running DB pods

  resourcePolicy:
    containerPolicies:
      - containerName: postgresql
        minAllowed:
          cpu: 1000m
          memory: 2Gi
        maxAllowed:
          cpu: 8000m
          memory: 16Gi
        controlledResources:
          - cpu
          - memory
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: redis-vpa
  namespace: $NAMESPACE
  labels:
    app: redis-cluster
spec:
  targetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: redis-cluster

  updatePolicy:
    updateMode: "Initial"

  resourcePolicy:
    containerPolicies:
      - containerName: redis
        minAllowed:
          cpu: 250m
          memory: 512Mi
        maxAllowed:
          cpu: 2000m
          memory: 4Gi
        controlledResources:
          - cpu
          - memory
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: chromadb-vpa
  namespace: $NAMESPACE
  labels:
    app: chromadb
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chromadb

  updatePolicy:
    updateMode: "Auto"

  resourcePolicy:
    containerPolicies:
      - containerName: chromadb
        minAllowed:
          cpu: 500m
          memory: 1Gi
        maxAllowed:
          cpu: 4000m
          memory: 16Gi
        controlledResources:
          - cpu
          - memory
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ollama-vpa
  namespace: $NAMESPACE
  labels:
    app: ollama
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ollama

  updatePolicy:
    updateMode: "Initial"  # GPU workloads should be updated carefully

  resourcePolicy:
    containerPolicies:
      - containerName: ollama
        minAllowed:
          cpu: 1000m
          memory: 4Gi
        maxAllowed:
          cpu: 8000m
          memory: 32Gi
        controlledResources:
          - cpu
          - memory
        # Don't auto-scale GPU resources
        controlledValues: RequestsAndLimits
EOF

echo -e "${GREEN}✓ VPA configurations created${NC}"
echo ""

# Create monitoring script
echo -e "${YELLOW}Creating VPA monitoring script...${NC}"

cat > /tmp/vpa-monitor.sh <<'SCRIPT'
#!/bin/bash

# VPA Recommendations Monitor

NAMESPACE="${NAMESPACE:-tars}"

echo "=== VPA Recommendations for T.A.R.S. ==="
echo ""

# Get all VPA resources
vpas=$(kubectl get vpa -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')

for vpa in $vpas; do
    echo "=== $vpa ==="
    echo ""

    # Get current recommendations
    kubectl get vpa $vpa -n $NAMESPACE -o jsonpath='
Target: {.spec.targetRef.kind}/{.spec.targetRef.name}
Update Mode: {.spec.updatePolicy.updateMode}

Current Resource Recommendations:
{range .status.recommendation.containerRecommendations[*]}
  Container: {.containerName}
    CPU:
      Target: {.target.cpu}
      Lower Bound: {.lowerBound.cpu}
      Upper Bound: {.upperBound.cpu}
      Uncapped Target: {.uncappedTarget.cpu}
    Memory:
      Target: {.target.memory}
      Lower Bound: {.lowerBound.memory}
      Upper Bound: {.upperBound.memory}
      Uncapped Target: {.uncappedTarget.memory}
{end}
'
    echo ""
    echo "---"
    echo ""
done

# Show VPA events
echo "=== Recent VPA Events ==="
kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | grep -i vpa | tail -20
echo ""
SCRIPT

chmod +x /tmp/vpa-monitor.sh
cp /tmp/vpa-monitor.sh /usr/local/bin/vpa-monitor || true

echo -e "${GREEN}✓ Monitoring script created${NC}"
echo ""

# Create VPA recommendation export script
cat > /tmp/vpa-export.sh <<'SCRIPT'
#!/bin/bash

# Export VPA recommendations to YAML

NAMESPACE="${NAMESPACE:-tars}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/vpa-recommendations}"

mkdir -p "$OUTPUT_DIR"

echo "Exporting VPA recommendations to $OUTPUT_DIR..."

vpas=$(kubectl get vpa -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')

for vpa in $vpas; do
    kubectl get vpa $vpa -n $NAMESPACE -o yaml > "$OUTPUT_DIR/${vpa}.yaml"
    echo "  ✓ Exported $vpa"
done

echo ""
echo "VPA recommendations exported to: $OUTPUT_DIR"
echo ""
echo "To apply recommendations, review the files and update your deployment manifests:"
echo "  1. Review: cat $OUTPUT_DIR/*.yaml"
echo "  2. Update deployment resource requests/limits based on recommendations"
echo "  3. Apply: kubectl apply -f <your-deployment.yaml>"
SCRIPT

chmod +x /tmp/vpa-export.sh
cp /tmp/vpa-export.sh /usr/local/bin/vpa-export || true

echo -e "${GREEN}✓ Export script created${NC}"
echo ""

# Summary
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}VPA Installation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

echo -e "${YELLOW}VPA Components Installed:${NC}"
kubectl get pods -n kube-system -l app.kubernetes.io/name=vpa
echo ""

echo -e "${YELLOW}VPA Configurations Created:${NC}"
kubectl get vpa -n $NAMESPACE
echo ""

echo -e "${YELLOW}Usage:${NC}"
echo -e "  • View recommendations: vpa-monitor"
echo -e "  • Export recommendations: vpa-export"
echo -e "  • Get specific VPA: kubectl get vpa <name> -n $NAMESPACE -o yaml"
echo ""

echo -e "${YELLOW}Update Modes:${NC}"
echo -e "  • Auto: VPA updates pod resource requests automatically"
echo -e "  • Recreate: VPA evicts and recreates pods with new recommendations"
echo -e "  • Initial: VPA sets resources only when pods are created"
echo -e "  • Off: VPA only provides recommendations without updating"
echo ""

echo -e "${YELLOW}Best Practices:${NC}"
echo -e "  1. Start with 'Initial' or 'Off' mode to observe recommendations"
echo -e "  2. For stateful workloads (databases), use 'Initial' mode"
echo -e "  3. For stateless workloads, 'Auto' mode is recommended"
echo -e "  4. Monitor VPA recommendations for 1-2 weeks before applying"
echo -e "  5. Don't use VPA and HPA together on the same resource"
echo ""

echo -e "${GREEN}Setup complete!${NC}"
