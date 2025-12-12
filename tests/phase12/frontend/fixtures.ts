/**
 * Playwright fixtures for T.A.R.S. frontend tests
 *
 * Provides:
 * - Authenticated page context
 * - Mock API responses
 * - Test users (viewer, developer, admin)
 * - Utility functions
 */

import { test as base, expect, Page } from '@playwright/test';

/**
 * Test user credentials
 */
export const TEST_USERS = {
  viewer: {
    username: 'viewer@test.com',
    password: 'viewer123',
    roles: ['viewer']
  },
  developer: {
    username: 'developer@test.com',
    password: 'developer123',
    roles: ['viewer', 'developer']
  },
  admin: {
    username: 'admin@test.com',
    password: 'admin123',
    roles: ['viewer', 'developer', 'admin']
  }
};

/**
 * Login helper function
 */
export async function login(page: Page, userType: 'viewer' | 'developer' | 'admin' = 'admin') {
  const user = TEST_USERS[userType];

  await page.goto('/login');
  await page.fill('input[name="username"]', user.username);
  await page.fill('input[name="password"]', user.password);
  await page.click('button[type="submit"]');

  // Wait for redirect to dashboard
  await page.waitForURL('/admin/dashboard', { timeout: 10000 });
}

/**
 * Logout helper function
 */
export async function logout(page: Page) {
  await page.click('[data-testid="user-menu"]');
  await page.click('[data-testid="logout-button"]');

  // Wait for redirect to login
  await page.waitForURL('/login', { timeout: 10000 });
}

/**
 * Mock API response helper
 */
export async function mockAPIResponse(
  page: Page,
  endpoint: string,
  response: any,
  method: string = 'GET'
) {
  await page.route(`**/admin${endpoint}`, async (route) => {
    if (route.request().method() === method) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(response)
      });
    } else {
      await route.continue();
    }
  });
}

/**
 * Wait for API call helper
 */
export async function waitForAPICall(page: Page, endpoint: string, method: string = 'GET') {
  return page.waitForRequest(
    (request) =>
      request.url().includes(endpoint) && request.method() === method,
    { timeout: 10000 }
  );
}

/**
 * Extended test fixture with authentication
 */
type TestFixtures = {
  authenticatedPage: Page;
  adminPage: Page;
  developerPage: Page;
  viewerPage: Page;
};

export const test = base.extend<TestFixtures>({
  /**
   * Authenticated page (admin role)
   */
  authenticatedPage: async ({ page }, use) => {
    await login(page, 'admin');
    await use(page);
    await logout(page);
  },

  /**
   * Admin page
   */
  adminPage: async ({ page }, use) => {
    await login(page, 'admin');
    await use(page);
  },

  /**
   * Developer page
   */
  developerPage: async ({ page }, use) => {
    await login(page, 'developer');
    await use(page);
  },

  /**
   * Viewer page
   */
  viewerPage: async ({ page }, use) => {
    await login(page, 'viewer');
    await use(page);
  },
});

export { expect };
