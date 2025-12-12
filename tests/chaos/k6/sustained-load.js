/**
 * T.A.R.S. Chaos Testing - Sustained Load Test (k6)
 *
 * Tests system behavior under sustained load.
 * Target: 100 RPS sustained for 10 minutes.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const jwtIssuanceDuration = new Trend('jwt_issuance_duration');
const jwtVerificationDuration = new Trend('jwt_verification_duration');
const apiKeyVerificationDuration = new Trend('api_key_verification_duration');
const agentReloadDuration = new Trend('agent_reload_duration');
const totalRequests = new Counter('total_requests');

// Test configuration
export const options = {
  scenarios: {
    sustained_load: {
      executor: 'constant-arrival-rate',
      rate: 100, // 100 RPS
      timeUnit: '1s',
      duration: '10m',
      preAllocatedVUs: 50,
      maxVUs: 200,
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.05'], // Less than 5% errors
    errors: ['rate<0.05'],
    jwt_issuance_duration: ['p(95)<100'],
    jwt_verification_duration: ['p(95)<10'],
  },
};

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const ADMIN_USERNAME = __ENV.ADMIN_USERNAME || 'admin';
const ADMIN_PASSWORD = __ENV.ADMIN_PASSWORD || 'admin123';

let authToken = null;

export function setup() {
  // Login once to get auth token
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
    console.error('No auth token available');
    errorRate.add(1);
    return;
  }

  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  // Mix of different endpoint calls (weighted distribution)
  const rand = Math.random();

  if (rand < 0.3) {
    // 30% - JWT status check (read-only)
    const res = http.get(`${BASE_URL}/admin/jwt/status`, { headers });
    const success = check(res, {
      'JWT status OK': (r) => r.status === 200,
    });
    errorRate.add(!success);
    totalRequests.add(1);

  } else if (rand < 0.5) {
    // 20% - List agents (read-only)
    const res = http.get(`${BASE_URL}/admin/agents`, { headers });
    const success = check(res, {
      'List agents OK': (r) => r.status === 200,
    });
    errorRate.add(!success);
    totalRequests.add(1);

  } else if (rand < 0.65) {
    // 15% - List API keys (read-only)
    const res = http.get(`${BASE_URL}/admin/api-keys`, { headers });
    const success = check(res, {
      'List API keys OK': (r) => r.status === 200,
    });
    errorRate.add(!success);
    totalRequests.add(1);

  } else if (rand < 0.80) {
    // 15% - Get AutoML trials (read-only)
    const res = http.get(`${BASE_URL}/admin/automl/trials`, { headers });
    const success = check(res, {
      'AutoML trials OK': (r) => r.status === 200,
    });
    errorRate.add(!success);
    totalRequests.add(1);

  } else if (rand < 0.90) {
    // 10% - Get system health (read-only)
    const res = http.get(`${BASE_URL}/admin/health`, { headers });
    const success = check(res, {
      'System health OK': (r) => r.status === 200,
    });
    errorRate.add(!success);
    totalRequests.add(1);

  } else {
    // 10% - Verify JWT token (simulates token verification)
    const startTime = Date.now();
    const res = http.get(`${BASE_URL}/admin/jwt/keys`, { headers });
    const duration = Date.now() - startTime;

    const success = check(res, {
      'JWT verification OK': (r) => r.status === 200,
    });

    errorRate.add(!success);
    jwtVerificationDuration.add(duration);
    totalRequests.add(1);
  }

  // Small sleep to avoid overwhelming the system
  sleep(0.1);
}

export function teardown(data) {
  console.log('Sustained load test completed');
}
