/**
 * T.A.R.S. Chaos Testing - JWT Stress Test (k6)
 *
 * Tests JWT issuance and verification under extreme load.
 * Simulates many concurrent users logging in and making authenticated requests.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const loginDuration = new Trend('login_duration');
const verificationDuration = new Trend('verification_duration');
const rotationDuration = new Trend('rotation_duration');
const loginSuccess = new Counter('login_success');
const loginFailure = new Counter('login_failure');

// Test configuration
export const options = {
  scenarios: {
    jwt_stress: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '1m', target: 50 },
        { duration: '3m', target: 200 },
        { duration: '1m', target: 50 },
        { duration: '1m', target: 0 },
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500'],
    errors: ['rate<0.05'],
    login_duration: ['p(95)<200'],
    verification_duration: ['p(95)<50'],
  },
};

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const ADMIN_USERNAME = __ENV.ADMIN_USERNAME || 'admin';
const ADMIN_PASSWORD = __ENV.ADMIN_PASSWORD || 'admin123';

export default function () {
  // 1. Login (JWT issuance)
  const loginStartTime = Date.now();
  const loginRes = http.post(`${BASE_URL}/auth/login`, JSON.stringify({
    username: ADMIN_USERNAME,
    password: ADMIN_PASSWORD,
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  const loginDur = Date.now() - loginStartTime;

  const loginOk = check(loginRes, {
    'Login successful': (r) => r.status === 200,
    'JWT token received': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.access_token && body.refresh_token;
      } catch {
        return false;
      }
    },
  });

  loginDuration.add(loginDur);

  if (!loginOk) {
    errorRate.add(1);
    loginFailure.add(1);
    sleep(1);
    return;
  }

  loginSuccess.add(1);

  const tokens = JSON.parse(loginRes.body);
  const accessToken = tokens.access_token;
  const refreshToken = tokens.refresh_token;

  const headers = {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json',
  };

  // 2. Multiple authenticated requests (JWT verification)
  for (let i = 0; i < 5; i++) {
    const verifyStartTime = Date.now();
    const res = http.get(`${BASE_URL}/admin/jwt/status`, { headers });
    const verifyDur = Date.now() - verifyStartTime;

    const success = check(res, {
      'JWT verification OK': (r) => r.status === 200,
    });

    verificationDuration.add(verifyDur);
    errorRate.add(!success);

    sleep(0.1);
  }

  // 3. Token refresh (10% of the time)
  if (Math.random() < 0.1) {
    const refreshRes = http.post(`${BASE_URL}/auth/refresh`, JSON.stringify({
      refresh_token: refreshToken,
    }), {
      headers: { 'Content-Type': 'application/json' },
    });

    const success = check(refreshRes, {
      'Token refresh OK': (r) => r.status === 200,
    });

    errorRate.add(!success);
  }

  sleep(1);
}

export function handleSummary(data) {
  const summary = {
    totalRequests: data.metrics.http_reqs.values.count,
    successfulLogins: data.metrics.login_success ? data.metrics.login_success.values.count : 0,
    failedLogins: data.metrics.login_failure ? data.metrics.login_failure.values.count : 0,
    errorRate: data.metrics.errors ? data.metrics.errors.values.rate : 0,
    avgLoginDuration: data.metrics.login_duration ? data.metrics.login_duration.values.avg : 0,
    avgVerificationDuration: data.metrics.verification_duration ? data.metrics.verification_duration.values.avg : 0,
    p95RequestDuration: data.metrics.http_req_duration ? data.metrics.http_req_duration.values['p(95)'] : 0,
  };

  console.log('\n=== JWT Stress Test Summary ===');
  console.log(`Total Requests: ${summary.totalRequests}`);
  console.log(`Successful Logins: ${summary.successfulLogins}`);
  console.log(`Failed Logins: ${summary.failedLogins}`);
  console.log(`Error Rate: ${(summary.errorRate * 100).toFixed(2)}%`);
  console.log(`Avg Login Duration: ${summary.avgLoginDuration.toFixed(2)}ms`);
  console.log(`Avg Verification Duration: ${summary.avgVerificationDuration.toFixed(2)}ms`);
  console.log(`P95 Request Duration: ${summary.p95RequestDuration.toFixed(2)}ms`);

  return {
    'stdout': JSON.stringify(summary, null, 2),
  };
}
