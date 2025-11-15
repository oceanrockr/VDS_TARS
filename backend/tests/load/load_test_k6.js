/**
 * T.A.R.S. K6 Load Testing Script
 * Phase 6: HTTP Performance Validation
 *
 * Measures:
 * - P50, P95, P99 latency
 * - Throughput (requests/sec)
 * - Error rate
 * - Cache hit ratio
 *
 * Usage:
 *   k6 run --vus 10 --duration 60s load_test_k6.js
 *   k6 run --vus 50 --duration 300s load_test_k6.js  # Stress test
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics
const cacheHits = new Counter('cache_hits');
const cacheMisses = new Counter('cache_misses');
const ragQueryLatency = new Trend('rag_query_latency', true);
const errorRate = new Rate('errors');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const TARGET_RPS = 200;  // Target requests per second
const DURATION = '60s';

// Test configuration
export const options = {
    stages: [
        { duration: '30s', target: 20 },   // Ramp up to 20 VUs
        { duration: '1m', target: 50 },    // Ramp up to 50 VUs
        { duration: '2m', target: 100 },   // Peak load: 100 VUs
        { duration: '1m', target: 50 },    // Ramp down to 50 VUs
        { duration: '30s', target: 0 },    // Ramp down to 0
    ],
    thresholds: {
        http_req_duration: ['p(50)<150', 'p(95)<250', 'p(99)<500'],  // Latency targets
        http_req_failed: ['rate<0.005'],  // Error rate < 0.5%
        errors: ['rate<0.005'],
        rag_query_latency: ['p(95)<250'],
    },
};

// Test data
const TEST_QUERIES = [
    "What is machine learning?",
    "Explain neural networks",
    "How does gradient descent work?",
    "What are transformers in NLP?",
    "Describe convolutional neural networks",
    "What is reinforcement learning?",
    "Explain attention mechanisms",
    "How do GANs work?",
    "What is transfer learning?",
    "Describe BERT model architecture",
];

let authToken = null;

// Setup function - runs once per VU
export function setup() {
    // Authenticate once
    const authPayload = JSON.stringify({
        client_id: `load_test_${Date.now()}_${Math.random()}`,
    });

    const authResponse = http.post(
        `${BASE_URL}/auth/authenticate`,
        authPayload,
        {
            headers: { 'Content-Type': 'application/json' },
        }
    );

    check(authResponse, {
        'authentication successful': (r) => r.status === 200,
    });

    const authData = JSON.parse(authResponse.body);
    return { token: authData.access_token };
}

// Main test function
export default function (data) {
    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${data.token}`,
    };

    // Test 1: Health check (10% of requests)
    if (Math.random() < 0.1) {
        const healthRes = http.get(`${BASE_URL}/health`);
        check(healthRes, {
            'health check status 200': (r) => r.status === 200,
            'health check has status field': (r) => JSON.parse(r.body).status === 'healthy',
        });
    }

    // Test 2: RAG query (70% of requests)
    if (Math.random() < 0.7) {
        const query = TEST_QUERIES[Math.floor(Math.random() * TEST_QUERIES.length)];
        const ragPayload = JSON.stringify({
            query: query,
            top_k: 5,
            rerank: true,
            include_sources: true,
        });

        const startTime = Date.now();
        const ragRes = http.post(`${BASE_URL}/rag/query`, ragPayload, { headers });
        const latency = Date.now() - startTime;

        const ragSuccess = check(ragRes, {
            'RAG query status 200': (r) => r.status === 200,
            'RAG query has answer': (r) => {
                try {
                    const body = JSON.parse(r.body);
                    return body.answer && body.answer.length > 0;
                } catch (e) {
                    return false;
                }
            },
            'RAG query latency < 3000ms': (r) => latency < 3000,
        });

        if (!ragSuccess) {
            errorRate.add(1);
        } else {
            errorRate.add(0);
        }

        ragQueryLatency.add(latency);

        // Check for cache hit (if response has metadata)
        try {
            const body = JSON.parse(ragRes.body);
            if (body.cached || body.from_cache) {
                cacheHits.add(1);
            } else {
                cacheMisses.add(1);
            }
        } catch (e) {
            // No cache metadata
        }
    }

    // Test 3: Analytics query (10% of requests)
    if (Math.random() < 0.1) {
        const analyticsRes = http.get(`${BASE_URL}/analytics/query-stats`, { headers });
        check(analyticsRes, {
            'analytics status 200': (r) => r.status === 200,
        });
    }

    // Test 4: Metrics endpoint (10% of requests)
    if (Math.random() < 0.1) {
        const metricsRes = http.get(`${BASE_URL}/metrics/system`, { headers });
        check(metricsRes, {
            'metrics status 200': (r) => r.status === 200,
        });
    }

    // Think time: simulate realistic user behavior
    sleep(Math.random() * 2 + 0.5);  // 0.5-2.5 seconds
}

// Teardown function
export function teardown(data) {
    console.log('Load test completed');
}

// Summary handler
export function handleSummary(data) {
    const cacheHitRatio = data.metrics.cache_hits.values.count /
        (data.metrics.cache_hits.values.count + data.metrics.cache_misses.values.count) * 100;

    console.log('\n=== T.A.R.S. Load Test Summary ===');
    console.log(`Total Requests: ${data.metrics.http_reqs.values.count}`);
    console.log(`Requests/sec: ${data.metrics.http_reqs.values.rate.toFixed(2)}`);
    console.log(`Error Rate: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%`);
    console.log(`P50 Latency: ${data.metrics.http_req_duration.values['p(50)'].toFixed(2)}ms`);
    console.log(`P95 Latency: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms`);
    console.log(`P99 Latency: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms`);
    console.log(`Cache Hit Ratio: ${cacheHitRatio.toFixed(2)}%`);
    console.log('=================================\n');

    return {
        'stdout': JSON.stringify(data, null, 2),
        'summary.json': JSON.stringify(data),
    };
}
