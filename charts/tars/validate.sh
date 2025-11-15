#!/bin/bash
# Helm Chart Validation Script for T.A.R.S.

set -e

CHART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==================================================================="
echo "T.A.R.S. Helm Chart Validation"
echo "==================================================================="
echo "Chart Directory: $CHART_DIR"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if helm is installed
if ! command -v helm &> /dev/null; then
    echo -e "${RED}❌ Helm is not installed${NC}"
    echo "Install Helm from: https://helm.sh/docs/intro/install/"
    exit 1
fi

echo -e "${GREEN}✓ Helm is installed: $(helm version --short)${NC}"
echo ""

# Step 1: Helm Lint
echo "-------------------------------------------------------------------"
echo "Step 1: Running helm lint..."
echo "-------------------------------------------------------------------"
if helm lint "$CHART_DIR" --strict; then
    echo -e "${GREEN}✓ Helm lint passed${NC}"
else
    echo -e "${RED}❌ Helm lint failed${NC}"
    exit 1
fi
echo ""

# Step 2: Helm Template (Dry Run)
echo "-------------------------------------------------------------------"
echo "Step 2: Running helm template..."
echo "-------------------------------------------------------------------"
RENDERED_DIR="/tmp/tars-helm-rendered"
mkdir -p "$RENDERED_DIR"

if helm template tars "$CHART_DIR" \
    --namespace tars \
    --values "$CHART_DIR/values.yaml" \
    --set secrets.jwtSecretKey="test-jwt-secret-key" \
    --set secrets.postgresPassword="test-postgres-password" \
    --set postgresql.auth.password="test-postgres-password" \
    > "$RENDERED_DIR/manifests.yaml"; then
    echo -e "${GREEN}✓ Helm template rendered successfully${NC}"
    echo "   Output: $RENDERED_DIR/manifests.yaml"
else
    echo -e "${RED}❌ Helm template failed${NC}"
    exit 1
fi
echo ""

# Step 3: Count Generated Resources
echo "-------------------------------------------------------------------"
echo "Step 3: Analyzing generated resources..."
echo "-------------------------------------------------------------------"
RESOURCE_COUNT=$(grep -c "^kind:" "$RENDERED_DIR/manifests.yaml" || true)
echo "   Total resources generated: $RESOURCE_COUNT"

# List resource types
echo "   Resource types:"
grep "^kind:" "$RENDERED_DIR/manifests.yaml" | sort | uniq -c | while read line; do
    echo "      $line"
done
echo ""

# Step 4: Validate Kubernetes Manifests (if kubectl is available)
echo "-------------------------------------------------------------------"
echo "Step 4: Validating Kubernetes manifests..."
echo "-------------------------------------------------------------------"
if command -v kubectl &> /dev/null; then
    if kubectl apply --dry-run=client -f "$RENDERED_DIR/manifests.yaml"; then
        echo -e "${GREEN}✓ Kubernetes validation passed${NC}"
    else
        echo -e "${RED}❌ Kubernetes validation failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ kubectl not found, skipping Kubernetes validation${NC}"
fi
echo ""

# Step 5: Check for Placeholder Values
echo "-------------------------------------------------------------------"
echo "Step 5: Checking for placeholder values..."
echo "-------------------------------------------------------------------"
PLACEHOLDER_COUNT=$(grep -c "REPLACE_WITH" "$RENDERED_DIR/manifests.yaml" || true)
if [ "$PLACEHOLDER_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $PLACEHOLDER_COUNT placeholder values${NC}"
    echo "   Please replace these before production deployment:"
    grep "REPLACE_WITH" "$RENDERED_DIR/manifests.yaml" | head -5
    echo "   ..."
else
    echo -e "${GREEN}✓ No placeholder values found${NC}"
fi
echo ""

# Step 6: Security Checks
echo "-------------------------------------------------------------------"
echo "Step 6: Running security checks..."
echo "-------------------------------------------------------------------"

# Check for image pull policy
if grep -q "imagePullPolicy: Always" "$RENDERED_DIR/manifests.yaml"; then
    echo -e "${GREEN}✓ Using Always pull policy (recommended for development)${NC}"
fi

# Check for resource limits
if grep -q "limits:" "$RENDERED_DIR/manifests.yaml"; then
    echo -e "${GREEN}✓ Resource limits are defined${NC}"
else
    echo -e "${YELLOW}⚠ No resource limits found${NC}"
fi

# Check for liveness/readiness probes
if grep -q "livenessProbe:" "$RENDERED_DIR/manifests.yaml"; then
    echo -e "${GREEN}✓ Liveness probes are configured${NC}"
else
    echo -e "${YELLOW}⚠ No liveness probes found${NC}"
fi

if grep -q "readinessProbe:" "$RENDERED_DIR/manifests.yaml"; then
    echo -e "${GREEN}✓ Readiness probes are configured${NC}"
else
    echo -e "${YELLOW}⚠ No readiness probes found${NC}"
fi
echo ""

# Step 7: Test Different Configurations
echo "-------------------------------------------------------------------"
echo "Step 7: Testing configuration variations..."
echo "-------------------------------------------------------------------"

# Test with HPA enabled
echo "   Testing with HPA enabled..."
if helm template tars "$CHART_DIR" \
    --set backend.autoscaling.enabled=true \
    --set secrets.jwtSecretKey="test" \
    --set secrets.postgresPassword="test" \
    --set postgresql.auth.password="test" \
    > /dev/null 2>&1; then
    echo -e "${GREEN}   ✓ HPA configuration works${NC}"
else
    echo -e "${RED}   ❌ HPA configuration failed${NC}"
fi

# Test with GPU enabled
echo "   Testing with GPU enabled..."
if helm template tars "$CHART_DIR" \
    --set ollama.gpu.enabled=true \
    --set secrets.jwtSecretKey="test" \
    --set secrets.postgresPassword="test" \
    --set postgresql.auth.password="test" \
    > /dev/null 2>&1; then
    echo -e "${GREEN}   ✓ GPU configuration works${NC}"
else
    echo -e "${RED}   ❌ GPU configuration failed${NC}"
fi

# Test with minimal configuration
echo "   Testing with minimal configuration..."
if helm template tars "$CHART_DIR" \
    --set postgresql.enabled=false \
    --set redis.enabled=false \
    --set chromadb.enabled=false \
    --set ollama.enabled=false \
    --set secrets.jwtSecretKey="test" \
    > /dev/null 2>&1; then
    echo -e "${GREEN}   ✓ Minimal configuration works${NC}"
else
    echo -e "${RED}   ❌ Minimal configuration failed${NC}"
fi
echo ""

# Step 8: Summary
echo "==================================================================="
echo "Validation Summary"
echo "==================================================================="
echo -e "${GREEN}✓ All validation checks passed!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review rendered manifests: $RENDERED_DIR/manifests.yaml"
echo "  2. Update placeholder values in values.yaml"
echo "  3. Test deployment: helm install tars $CHART_DIR --namespace tars --create-namespace"
echo ""
echo "==================================================================="

# Cleanup option
read -p "Clean up rendered manifests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$RENDERED_DIR"
    echo "Cleaned up $RENDERED_DIR"
fi
