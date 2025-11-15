#!/bin/bash

# Cluster Autoscaler Setup Script for T.A.R.S.
# Automatically scales cluster nodes based on resource demands

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-kube-system}"
CLUSTER_NAME="${CLUSTER_NAME:-tars-cluster}"
CLOUD_PROVIDER="${CLOUD_PROVIDER:-aws}"  # aws, gcp, azure, or generic
MIN_NODES="${MIN_NODES:-3}"
MAX_NODES="${MAX_NODES:-20}"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Cluster Autoscaler Setup for T.A.R.S.${NC}"
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

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"
echo ""

# Create RBAC resources
echo -e "${YELLOW}Creating RBAC resources...${NC}"

cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cluster-autoscaler
  namespace: $NAMESPACE
  labels:
    app: cluster-autoscaler
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-autoscaler
  labels:
    app: cluster-autoscaler
rules:
  - apiGroups: [""]
    resources: ["events", "endpoints"]
    verbs: ["create", "patch"]
  - apiGroups: [""]
    resources: ["pods/eviction"]
    verbs: ["create"]
  - apiGroups: [""]
    resources: ["pods/status"]
    verbs: ["update"]
  - apiGroups: [""]
    resources: ["endpoints"]
    resourceNames: ["cluster-autoscaler"]
    verbs: ["get", "update"]
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["watch", "list", "get", "update"]
  - apiGroups: [""]
    resources:
      - "namespaces"
      - "pods"
      - "services"
      - "replicationcontrollers"
      - "persistentvolumeclaims"
      - "persistentvolumes"
    verbs: ["watch", "list", "get"]
  - apiGroups: ["extensions", "apps"]
    resources: ["daemonsets", "replicasets", "statefulsets"]
    verbs: ["watch", "list", "get"]
  - apiGroups: ["policy"]
    resources: ["poddisruptionbudgets"]
    verbs: ["watch", "list"]
  - apiGroups: ["apps"]
    resources: ["statefulsets", "replicasets", "daemonsets"]
    verbs: ["watch", "list", "get"]
  - apiGroups: ["storage.k8s.io"]
    resources: ["storageclasses", "csinodes", "csidrivers", "csistoragecapacities"]
    verbs: ["watch", "list", "get"]
  - apiGroups: ["batch"]
    resources: ["jobs", "cronjobs"]
    verbs: ["watch", "list", "get"]
  - apiGroups: ["coordination.k8s.io"]
    resources: ["leases"]
    verbs: ["create"]
  - apiGroups: ["coordination.k8s.io"]
    resourceNames: ["cluster-autoscaler"]
    resources: ["leases"]
    verbs: ["get", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: cluster-autoscaler
  namespace: $NAMESPACE
  labels:
    app: cluster-autoscaler
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["create", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    resourceNames: ["cluster-autoscaler-status", "cluster-autoscaler-priority-expander"]
    verbs: ["delete", "get", "update", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cluster-autoscaler
  labels:
    app: cluster-autoscaler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-autoscaler
subjects:
  - kind: ServiceAccount
    name: cluster-autoscaler
    namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: cluster-autoscaler
  namespace: $NAMESPACE
  labels:
    app: cluster-autoscaler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: cluster-autoscaler
subjects:
  - kind: ServiceAccount
    name: cluster-autoscaler
    namespace: $NAMESPACE
EOF

echo -e "${GREEN}✓ RBAC resources created${NC}"
echo ""

# Create Cluster Autoscaler deployment
echo -e "${YELLOW}Creating Cluster Autoscaler deployment...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: $NAMESPACE
  labels:
    app: cluster-autoscaler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8085"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: cluster-autoscaler
      priorityClassName: system-cluster-critical

      containers:
        - name: cluster-autoscaler
          image: registry.k8s.io/autoscaling/cluster-autoscaler:v1.28.2
          imagePullPolicy: IfNotPresent

          command:
            - ./cluster-autoscaler
            - --v=4
            - --stderrthreshold=info
            - --cloud-provider=$CLOUD_PROVIDER
            - --skip-nodes-with-local-storage=false
            - --expander=least-waste
            - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/$CLUSTER_NAME
            - --balance-similar-node-groups
            - --skip-nodes-with-system-pods=false
            - --scale-down-enabled=true
            - --scale-down-delay-after-add=10m
            - --scale-down-unneeded-time=10m
            - --scale-down-utilization-threshold=0.5
            - --max-node-provision-time=15m
            - --max-graceful-termination-sec=600
            - --new-pod-scale-up-delay=0s
            - --scan-interval=10s

          env:
            # AWS specific configuration
            - name: AWS_REGION
              value: "us-east-1"

          ports:
            - name: metrics
              containerPort: 8085
              protocol: TCP

          resources:
            limits:
              cpu: 500m
              memory: 512Mi
            requests:
              cpu: 100m
              memory: 128Mi

          livenessProbe:
            httpGet:
              path: /health-check
              port: 8085
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          readinessProbe:
            httpGet:
              path: /health-check
              port: 8085
            initialDelaySeconds: 10
            periodSeconds: 5
            timeoutSeconds: 3

          volumeMounts:
            - name: ssl-certs
              mountPath: /etc/ssl/certs/ca-certificates.crt
              readOnly: true

      volumes:
        - name: ssl-certs
          hostPath:
            path: /etc/ssl/certs/ca-certificates.crt
            type: File

      nodeSelector:
        kubernetes.io/os: linux

      tolerations:
        - effect: NoSchedule
          key: node-role.kubernetes.io/master
        - effect: NoSchedule
          key: node-role.kubernetes.io/control-plane

      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                  - key: node-role.kubernetes.io/control-plane
                    operator: Exists
EOF

echo -e "${GREEN}✓ Deployment created${NC}"
echo ""

# Create Service for metrics
echo -e "${YELLOW}Creating service for Prometheus metrics...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: cluster-autoscaler
  namespace: $NAMESPACE
  labels:
    app: cluster-autoscaler
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8085"
spec:
  type: ClusterIP
  ports:
    - name: metrics
      port: 8085
      targetPort: 8085
      protocol: TCP
  selector:
    app: cluster-autoscaler
EOF

echo -e "${GREEN}✓ Service created${NC}"
echo ""

# Create ServiceMonitor for Prometheus
echo -e "${YELLOW}Creating ServiceMonitor for Prometheus...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: cluster-autoscaler
  namespace: $NAMESPACE
  labels:
    app: cluster-autoscaler
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
EOF

echo -e "${GREEN}✓ ServiceMonitor created${NC}"
echo ""

# Create PodDisruptionBudget
echo -e "${YELLOW}Creating PodDisruptionBudget...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: cluster-autoscaler
  namespace: $NAMESPACE
  labels:
    app: cluster-autoscaler
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
EOF

echo -e "${GREEN}✓ PodDisruptionBudget created${NC}"
echo ""

# Create ConfigMap for autoscaler configuration
echo -e "${YELLOW}Creating autoscaler configuration...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-priority-expander
  namespace: $NAMESPACE
  labels:
    app: cluster-autoscaler
data:
  priorities: |
    10:
      - .*-spot-.*
    50:
      - .*-ondemand-.*
    100:
      - .*-gpu-.*
EOF

echo -e "${GREEN}✓ Configuration created${NC}"
echo ""

# Create PrometheusRule for alerts
echo -e "${YELLOW}Creating Prometheus alerts...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: cluster-autoscaler-alerts
  namespace: $NAMESPACE
  labels:
    app: cluster-autoscaler
spec:
  groups:
    - name: cluster_autoscaler_alerts
      interval: 30s
      rules:
        - alert: ClusterAutoscalerDown
          expr: up{job="cluster-autoscaler"} == 0
          for: 5m
          labels:
            severity: critical
            component: cluster-autoscaler
          annotations:
            summary: "Cluster Autoscaler is down"
            description: "Cluster Autoscaler has been down for more than 5 minutes"

        - alert: ClusterAutoscalerErrors
          expr: rate(cluster_autoscaler_errors_total[5m]) > 0.1
          for: 5m
          labels:
            severity: warning
            component: cluster-autoscaler
          annotations:
            summary: "Cluster Autoscaler has high error rate"
            description: "Error rate is {{ \$value }} errors/sec"

        - alert: ClusterAutoscalerFailedScaleUp
          expr: rate(cluster_autoscaler_failed_scale_ups_total[5m]) > 0.1
          for: 10m
          labels:
            severity: warning
            component: cluster-autoscaler
          annotations:
            summary: "Cluster Autoscaler failed to scale up"
            description: "Failed scale-up rate is {{ \$value }}/sec"

        - alert: ClusterAutoscalerUnschedulablePods
          expr: cluster_autoscaler_unschedulable_pods_count > 5
          for: 10m
          labels:
            severity: warning
            component: cluster-autoscaler
          annotations:
            summary: "High number of unschedulable pods"
            description: "{{ \$value }} pods cannot be scheduled"

        - alert: ClusterNearMaxSize
          expr: cluster_autoscaler_nodes_count >= ($MAX_NODES * 0.9)
          for: 5m
          labels:
            severity: warning
            component: cluster-autoscaler
          annotations:
            summary: "Cluster is near maximum size"
            description: "Cluster has {{ \$value }} nodes (max: $MAX_NODES)"
EOF

echo -e "${GREEN}✓ Alerts created${NC}"
echo ""

# Create monitoring script
cat > /tmp/autoscaler-monitor.sh <<'SCRIPT'
#!/bin/bash

# Cluster Autoscaler Monitor

NAMESPACE="${NAMESPACE:-kube-system}"

echo "=== Cluster Autoscaler Status ==="
echo ""

# Pod status
echo "Pod Status:"
kubectl get pods -n $NAMESPACE -l app=cluster-autoscaler
echo ""

# Logs (last 50 lines)
echo "Recent Logs:"
kubectl logs -n $NAMESPACE -l app=cluster-autoscaler --tail=50
echo ""

# Metrics
echo "Cluster Autoscaler Metrics:"
kubectl exec -n $NAMESPACE -l app=cluster-autoscaler -- wget -q -O- http://localhost:8085/metrics | grep -E "cluster_autoscaler_(nodes|unschedulable|failed)"
echo ""

# Node count
echo "Node Count:"
kubectl get nodes --no-headers | wc -l
echo ""

# Unschedulable pods
echo "Unschedulable Pods:"
kubectl get pods --all-namespaces --field-selector=status.phase==Pending
SCRIPT

chmod +x /tmp/autoscaler-monitor.sh
cp /tmp/autoscaler-monitor.sh /usr/local/bin/autoscaler-monitor || true

echo -e "${GREEN}✓ Monitoring script created${NC}"
echo ""

# Wait for autoscaler pod
echo -e "${YELLOW}Waiting for Cluster Autoscaler pod to be ready...${NC}"
kubectl wait --for=condition=ready pod \
    -l app=cluster-autoscaler \
    -n $NAMESPACE \
    --timeout=300s || true

echo -e "${GREEN}✓ Cluster Autoscaler is ready${NC}"
echo ""

# Summary
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Cluster Autoscaler Installation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Cluster Name: $CLUSTER_NAME"
echo -e "  Cloud Provider: $CLOUD_PROVIDER"
echo -e "  Min Nodes: $MIN_NODES"
echo -e "  Max Nodes: $MAX_NODES"
echo ""

echo -e "${YELLOW}Status:${NC}"
kubectl get deployment cluster-autoscaler -n $NAMESPACE
echo ""

echo -e "${YELLOW}Usage:${NC}"
echo -e "  • View logs: kubectl logs -n $NAMESPACE -l app=cluster-autoscaler -f"
echo -e "  • Monitor status: autoscaler-monitor"
echo -e "  • View metrics: kubectl port-forward -n $NAMESPACE svc/cluster-autoscaler 8085:8085"
echo -e "    Then: curl http://localhost:8085/metrics"
echo ""

echo -e "${YELLOW}Key Metrics:${NC}"
echo -e "  • cluster_autoscaler_nodes_count: Current node count"
echo -e "  • cluster_autoscaler_unschedulable_pods_count: Pods waiting for scale-up"
echo -e "  • cluster_autoscaler_failed_scale_ups_total: Failed scale-up attempts"
echo -e "  • cluster_autoscaler_scaled_up_nodes_total: Successful node additions"
echo ""

echo -e "${GREEN}Setup complete!${NC}"
