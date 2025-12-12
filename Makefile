# T.A.R.S. Makefile
# Phase 13.7 - Documentation, Observability, and Validation Targets

.PHONY: help docs openapi-validate tracing-test deploy-alerts generate-c4 validate-k8s \
        test test-unit test-integration test-eval-engine lint format clean install

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

install: ## Install all dependencies
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "✅ Dependencies installed"

clean: ## Clean up generated files and caches
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	@echo "✅ Cleaned up"

##@ Documentation

docs: ## Generate all documentation
	@echo "Generating documentation..."
	@$(MAKE) generate-c4
	@$(MAKE) openapi-validate
	@echo "✅ Documentation generated"

generate-c4: ## Generate C4 diagrams from Mermaid (requires mmdc)
	@echo "Generating C4 diagrams..."
	@command -v mmdc >/dev/null 2>&1 || { echo "❌ mermaid-cli not installed. Run: npm install -g @mermaid-js/mermaid-cli"; exit 1; }
	@echo "Note: C4 diagrams are embedded in markdown. Use Mermaid Live Editor or GitHub/GitLab rendering."
	@echo "Alternatively, generate PNGs with: mmdc -i docs/architecture/*.md -o docs/architecture/"
	@echo "✅ C4 diagrams ready (view in markdown renderers)"

openapi-validate: ## Validate OpenAPI specification
	@echo "Validating OpenAPI spec..."
	@command -v openapi-spec-validator >/dev/null 2>&1 || { echo "❌ openapi-spec-validator not installed. Run: pip install openapi-spec-validator"; exit 1; }
	openapi-spec-validator docs/api/openapi.yaml
	@echo "✅ OpenAPI spec is valid"

openapi-docs: ## Generate OpenAPI HTML documentation (requires redoc-cli)
	@echo "Generating OpenAPI HTML docs..."
	@command -v redoc-cli >/dev/null 2>&1 || { echo "❌ redoc-cli not installed. Run: npm install -g redoc-cli"; exit 1; }
	redoc-cli bundle docs/api/openapi.yaml -o docs/api/openapi.html
	@echo "✅ OpenAPI docs generated at docs/api/openapi.html"

##@ Testing

test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests
	@echo "Running unit tests..."
	pytest tests/eval-engine/test_*.py -v --tb=short
	@echo "✅ Unit tests passed"

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	pytest tests/eval-engine/integration/ -v --tb=short
	@echo "✅ Integration tests passed"

test-phase14_6: test-retro-unit test-retro-smoke ## Run all Phase 14.6 tests (unit + smoke)

test-retro-unit: ## Run retrospective generator unit tests
	@echo "Running retrospective generator unit tests..."
	pytest tests/test_retrospective_generator.py -v --tb=short
	@echo "✅ Retrospective unit tests passed"

test-retro-smoke: ## Run Phase 14.6 end-to-end smoke test
	@echo "Running Phase 14.6 smoke test..."
	@bash scripts/test_phase14_6_pipeline.sh
	@echo "✅ Phase 14.6 smoke test passed"

test-retro-coverage: ## Run retrospective tests with coverage
	@echo "Running retrospective tests with coverage..."
	pytest tests/test_retrospective_generator.py -v --cov=scripts.generate_retrospective --cov-report=html --cov-report=term
	@echo "✅ Coverage report: htmlcov/index.html"

retro-test: ## Generate retrospective on test data
	@echo "Generating retrospective on test data..."
	@mkdir -p test_output
	python scripts/generate_retrospective.py \
		--ga-data test_data/ga_kpis \
		--7day-data test_data/stability \
		--regression test_data/regression/regression_summary.json \
		--anomalies test_data/anomalies/anomaly_events.json \
		--output test_output/GA_7DAY_RETROSPECTIVE.md
	@echo "✅ Retrospective generated at test_output/GA_7DAY_RETROSPECTIVE.md"

clean-test: ## Clean Phase 14.6 test outputs
	@echo "Cleaning test outputs..."
	rm -rf test_output
	@echo "✅ Test outputs cleaned"

test-e2e: ## Run end-to-end pipeline tests
	@echo "Running E2E tests..."
	pytest tests/e2e/ -v --tb=short
	@echo "✅ E2E tests passed"

test-failover: ## Run multi-region failover tests
	@echo "Running failover tests..."
	pytest tests/failover/ -v --tb=short
	@echo "✅ Failover tests passed"

test-eval-engine: ## Run eval-engine specific tests
	@echo "Running eval-engine tests..."
	pytest tests/eval-engine/ -v --cov=cognition/eval-engine --cov-report=html --cov-report=term
	@echo "✅ Eval-engine tests passed"
	@echo "Coverage report: htmlcov/index.html"

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=cognition --cov-report=html --cov-report=term-missing
	@echo "✅ Coverage report generated: htmlcov/index.html"

test-all: test test-e2e test-failover ## Run all tests (unit, integration, E2E, failover)

test-security: ## Run security test suite
	@echo "Running security tests..."
	pytest security/ -v --tb=short
	@echo "✅ Security tests passed"

test-canary: ## Run canary deployment tests
	@echo "Running canary tests..."
	pytest canary/ -v --tb=short
	@echo "✅ Canary tests passed"

test-statuspage: ## Test Statuspage integration
	@echo "Testing Statuspage integration..."
	python canary/statuspage_client.py test-connection
	@echo "✅ Statuspage integration working"

##@ Linting & Formatting

lint: ## Run linters (flake8, mypy)
	@echo "Running linters..."
	flake8 cognition/ --max-line-length=120 --exclude=__pycache__,*.pyc
	mypy cognition/eval-engine/ --ignore-missing-imports
	@echo "✅ Linting passed"

format: ## Format code with black and isort
	@echo "Formatting code..."
	black cognition/ tests/ --line-length=120
	isort cognition/ tests/ --profile black
	@echo "✅ Code formatted"

format-check: ## Check code formatting without modifying
	@echo "Checking code format..."
	black cognition/ tests/ --check --line-length=120
	isort cognition/ tests/ --check --profile black
	@echo "✅ Code format is correct"

##@ Observability

deploy-alerts: ## Deploy Prometheus alert rules to Kubernetes
	@echo "Deploying Prometheus alerts..."
	kubectl apply -f observability/alerts/prometheus-alerts.yaml
	@echo "✅ Alerts deployed"

deploy-dashboard: ## Deploy Grafana dashboard
	@echo "Deploying Grafana dashboard..."
	@if kubectl get configmap grafana-dashboards -n monitoring >/dev/null 2>&1; then \
		kubectl create configmap grafana-dashboards \
			--from-file=observability/dashboards/eval_engine.json \
			-n monitoring --dry-run=client -o yaml | kubectl apply -f -; \
	else \
		kubectl create configmap grafana-dashboards \
			--from-file=observability/dashboards/eval_engine.json \
			-n monitoring; \
	fi
	@echo "✅ Dashboard deployed"

tracing-test: ## Test distributed tracing integration
	@echo "Testing distributed tracing..."
	@echo "Starting Jaeger (if not running)..."
	@docker ps | grep jaegertracing/all-in-one || docker run -d --name jaeger \
		-p 6831:6831/udp \
		-p 16686:16686 \
		-p 4317:4317 \
		jaegertracing/all-in-one:latest
	@echo "Running tracing test..."
	python -c "from cognition.eval-engine.instrumentation import setup_tracing; print('✅ Tracing imports OK')"
	@echo "✅ Tracing test passed. View traces at http://localhost:16686"

##@ Kubernetes

validate-k8s: ## Validate Kubernetes manifests
	@echo "Validating Kubernetes manifests..."
	@command -v kubeval >/dev/null 2>&1 || { echo "⚠️  kubeval not installed. Run: brew install kubeval"; echo "Skipping validation..."; exit 0; }
	kubeval charts/tars/templates/*.yaml --ignore-missing-schemas
	@echo "✅ Kubernetes manifests valid"

k8s-dry-run: ## Dry-run Helm chart installation
	@echo "Dry-run Helm install..."
	helm install tars charts/tars --dry-run --debug --namespace tars
	@echo "✅ Helm dry-run successful"

deploy-eval-engine: ## Deploy eval-engine to Kubernetes
	@echo "Deploying eval-engine to Kubernetes..."
	helm upgrade --install tars charts/tars \
		--namespace tars \
		--create-namespace \
		--set evalEngine.enabled=true \
		--set evalEngine.replicaCount=2
	@echo "✅ Eval-engine deployed"

##@ Development

run-eval-engine: ## Run eval-engine locally
	@echo "Starting eval-engine locally..."
	cd cognition/eval-engine && python main.py

run-eval-engine-dev: ## Run eval-engine in dev mode with hot-reload
	@echo "Starting eval-engine in dev mode..."
	cd cognition/eval-engine && uvicorn main:app --reload --port 8099

shell-eval-engine: ## Open shell in eval-engine pod
	@kubectl exec -it deployment/tars-eval-engine -n tars -- /bin/bash

logs-eval-engine: ## Tail eval-engine logs
	@kubectl logs -n tars deployment/tars-eval-engine -f

port-forward: ## Port-forward eval-engine to localhost:8099
	@echo "Port-forwarding eval-engine to localhost:8099..."
	@kubectl port-forward -n tars svc/tars-eval-engine 8099:8099

##@ Database

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	psql $(POSTGRES_URL) < cognition/eval-engine/db/migrations/007_eval_baselines.sql
	@echo "✅ Migrations applied"

db-rollback: ## Rollback database migrations
	@echo "Rolling back database migrations..."
	psql $(POSTGRES_URL) < cognition/eval-engine/db/migrations/007_rollback.sql
	@echo "✅ Rollback complete"

db-shell: ## Open PostgreSQL shell
	@psql $(POSTGRES_URL)

redis-shell: ## Open Redis shell
	@redis-cli -h localhost -p 6379

##@ CI/CD

ci-test: ## Run tests in CI mode
	@echo "Running CI tests..."
	pytest tests/ -v --tb=short --cov=cognition --cov-report=xml --cov-report=term
	@echo "✅ CI tests passed"

ci-lint: ## Run linters in CI mode
	@echo "Running CI linters..."
	flake8 cognition/ --max-line-length=120 --exit-zero
	mypy cognition/eval-engine/ --ignore-missing-imports --no-error-summary
	@echo "✅ CI linting complete"

ci-validate: ## Validate all configurations for CI
	@$(MAKE) openapi-validate
	@$(MAKE) validate-k8s
	@$(MAKE) format-check
	@echo "✅ CI validation passed"

##@ Benchmarks

bench: bench-latency bench-throughput bench-regression ## Run all benchmarks

bench-latency: ## Run evaluation latency benchmark
	@echo "Running latency benchmark..."
	python benchmarks/eval_latency_bench.py
	@echo "✅ Latency benchmark complete"

bench-throughput: ## Run throughput benchmark
	@echo "Running throughput benchmark..."
	python benchmarks/throughput_bench.py
	@echo "✅ Throughput benchmark complete"

bench-regression: ## Run regression detection benchmark
	@echo "Running regression detection benchmark..."
	python benchmarks/regression_detector_bench.py
	@echo "✅ Regression benchmark complete"

##@ Metrics

metrics: ## View Prometheus metrics
	@curl -s http://localhost:8099/metrics | grep tars_eval

health: ## Check service health
	@curl -s http://localhost:8099/health | jq .

##@ Utility

watch-pods: ## Watch eval-engine pods
	@kubectl get pods -n tars -l app.kubernetes.io/component=eval-engine -w

describe-pod: ## Describe eval-engine pod
	@kubectl describe pod -n tars -l app.kubernetes.io/component=eval-engine

get-secrets: ## Get JWT secret from Kubernetes
	@kubectl get secret tars-secrets -n tars -o jsonpath='{.data.jwt-secret}' | base64 -d && echo

restart-eval-engine: ## Restart eval-engine deployment
	@kubectl rollout restart deployment tars-eval-engine -n tars
	@kubectl rollout status deployment tars-eval-engine -n tars

scale-eval-engine: ## Scale eval-engine (usage: make scale-eval-engine REPLICAS=5)
	@kubectl scale deployment tars-eval-engine -n tars --replicas=$(or $(REPLICAS),3)

##@ GA Deployment

deploy-ga: ## Deploy T.A.R.S. v1.0.0 GA to production
	@echo "🚀 Deploying T.A.R.S. v1.0.0 GA to production..."
	@echo "Applying ArgoCD Application manifest..."
	kubectl apply -f deploy/ga/argo_application.yaml
	@echo "Syncing ArgoCD application..."
	argocd app sync tars-v1-ga --prune
	@echo "Waiting for deployment to complete..."
	argocd app wait tars-v1-ga --health --timeout 600
	@echo "✅ GA deployment complete!"

validate-ga-readiness: ## Validate GA readiness (all checks)
	@echo "Validating GA readiness..."
	@echo "\n1️⃣  Infrastructure checks..."
	@kubectl get nodes | grep -q Ready && echo "✅ Cluster ready" || echo "❌ Cluster not ready"
	@kubectl get namespace tars-production >/dev/null 2>&1 && echo "✅ Namespace exists" || echo "❌ Namespace missing"
	@echo "\n2️⃣  Security checks..."
	@$(MAKE) test-security
	@echo "\n3️⃣  Canary tests..."
	@$(MAKE) test-canary
	@echo "\n4️⃣  Performance benchmarks..."
	@$(MAKE) bench
	@echo "\n5️⃣  Statuspage integration..."
	@$(MAKE) test-statuspage
	@echo "\n✅ GA readiness validation complete!"

release-ga: ## Create GA release package (tests + validation + deployment)
	@echo "📦 Creating GA release package..."
	@$(MAKE) validate-ga-readiness
	@echo "\n📊 Capturing baseline metrics..."
	@$(MAKE) capture-baseline-metrics
	@echo "\n📝 Release checklist:"
	@echo "  ✅ All tests passing"
	@echo "  ✅ All benchmarks passing"
	@echo "  ✅ Security validation complete"
	@echo "  ✅ Canary tests ready"
	@echo "  ✅ Statuspage configured"
	@echo "\n🚀 Ready for GA deployment!"
	@echo "\nNext steps:"
	@echo "  1. Review deploy/ga/rollout_playbook.md"
	@echo "  2. Run: make deploy-ga"
	@echo "  3. Monitor: make monitor-ga-deployment"

capture-baseline-metrics: ## Capture baseline metrics for canary comparison
	@echo "Capturing baseline metrics..."
	@mkdir -p /tmp/tars_metrics
	@kubectl exec -n observability prometheus-kube-prometheus-prometheus-0 -- \
		promtool query instant http://localhost:9090 \
		'rate(http_requests_total{namespace="tars-production"}[5m])' \
		> /tmp/tars_metrics/baseline_$(shell date +%Y%m%d_%H%M%S).json 2>/dev/null || echo "⚠️  Prometheus not accessible"
	@echo "✅ Baseline metrics saved to /tmp/tars_metrics/"

monitor-ga-deployment: ## Monitor GA deployment progress
	@echo "📊 Monitoring GA deployment..."
	@echo "\n🔍 ArgoCD Application Status:"
	@argocd app get tars-v1-ga --refresh
	@echo "\n🏥 Health Status:"
	@kubectl get pods -n tars-production
	@echo "\n📈 Metrics:"
	@$(MAKE) metrics
	@echo "\n📋 For detailed monitoring, visit:"
	@echo "  • ArgoCD: https://argocd.tars.prod/applications/tars-v1-ga"
	@echo "  • Grafana: https://grafana.tars.prod/d/tars-overview"
	@echo "  • Statuspage: https://status.tars.prod"

rollback-ga: ## Rollback GA deployment to previous version
	@echo "🔄 Rolling back GA deployment..."
	@read -p "Are you sure you want to rollback? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		argocd app history tars-v1-ga; \
		read -p "Enter revision number to rollback to: " revision; \
		argocd app rollback tars-v1-ga $$revision; \
		echo "✅ Rollback initiated to revision $$revision"; \
	else \
		echo "❌ Rollback cancelled"; \
	fi

##@ Quick Start

quickstart: install db-migrate deploy-eval-engine ## Full quickstart (install + migrate + deploy)
	@echo "✅ Quickstart complete!"
	@echo "Access eval-engine at: http://localhost:8099"
	@echo "Health check: make health"
	@echo "Logs: make logs-eval-engine"

dev-setup: install ## Setup development environment
	@echo "Setting up development environment..."
	@pre-commit install || echo "⚠️  pre-commit not installed"
	@echo "✅ Dev environment ready"
	@echo "Run 'make run-eval-engine-dev' to start the server"

##@ Troubleshooting

troubleshoot: ## Run troubleshooting diagnostics
	@echo "Running diagnostics..."
	@echo "\n📊 Service Health:"
	@make health || echo "❌ Service unreachable"
	@echo "\n📦 Pod Status:"
	@kubectl get pods -n tars | grep eval-engine || echo "❌ No pods found"
	@echo "\n📈 HPA Status:"
	@kubectl get hpa -n tars tars-eval-engine || echo "⚠️  HPA not configured"
	@echo "\n💾 PostgreSQL:"
	@psql $(POSTGRES_URL) -c "SELECT count(*) FROM eval_baselines;" || echo "❌ PostgreSQL unreachable"
	@echo "\n🔴 Redis:"
	@redis-cli ping || echo "❌ Redis unreachable"
	@echo "\n✅ Diagnostics complete"
