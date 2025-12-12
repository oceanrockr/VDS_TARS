/**
 * T.A.R.S. Chaos Testing - Spike Load Test (k6)
 *
 * Tests system behavior under sudden traffic spikes.
 * Simulates normal load with periodic spikes to 500 RPS.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const requestDuration = new Trend('request_duration');
const totalRequests = new Counter('total_requests');
const rateLimitHits = new Counter('rate_limit_hits');

// Test configuration
export const options = {
  scenarios: {
    spike_test: {
      executor: 'ramping-arrival-rate',
      startRate: 50,
      timeUnit: '1s',
      preAllocatedVUs: 50,
      maxVUs: 500,
      stages: [
        { duration: '2m', target: 50 },   // Normal load
        { duration: '30s', target: 500 }, // Spike!
        { duration: '1m', target: 50 },   // Recovery
        { duration: '30s', target: 500 }, // Second spike
        { duration: '2m', target: 50 },   // Cool down
      ],
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],
    http_req_failed: ['rate<0.15'], // Allow up to 15% errors during spikes
    errors: ['rate<0.15'],
  },
};

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const ADMIN_USERNAME = __ENV.ADMIN_USERNAME || 'admin';
const ADMIN_PASSWORD = __ENV.ADMIN_PASSWORD || 'admin123';

export function setup() {
  // Login to get auth token
  const loginRes = http.post(`${BASE_URL}/auth/login`, JSON.stringify({
    username: ADMIN_USERNAME,
    password: ADMIN_PASSWORD,
  }), {
    headers: { 'Content-Type': 'application/json' },
  });

  if (loginRes.status === 200) {
    const body = JSON.parse(loginRes.body);
    return { token: body.access_token };
  } else {
    console.error('Failed to login during setup');
    return { token: null };
  }
}

export default function (data) {
  const token = data.token;
  if (!token) {
    errorRate.add(1);
    return;
  }

  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  // Stress test endpoint (most expensive operation)
  const startTime = Date.now();
  const res = http.get(`${BASE_URL}/admin/agents`, { headers });
  const duration = Date.now() - startTime;

  const success = check(res, {
    'Status is 200 or 429': (r) => r.status === 200 || r.status === 429,
  });

  if (res.status === 429) {
    rateLimitHits.add(1);
  }

  errorRate.add(!success);
  requestDuration.add(duration);
  totalRequests.add(1);

  sleep(0.05);
}

export function handleSummary(data) {
  return {
    'stdout': JSON.stringify(data, null, 2),
  };
}
