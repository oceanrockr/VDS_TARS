/**
 * Authentication E2E tests
 *
 * Tests:
 * - Valid login
 * - Invalid login
 * - Expired token rejection
 * - Logout clears token
 * - Unauthorized access redirects to login
 */

import { test, expect, Page } from '@playwright/test';
import { TEST_USERS, login, logout } from './fixtures';

test.describe('Authentication', () => {
  test('should login successfully with valid credentials', async ({ page }) => {
    await page.goto('/login');

    // Fill login form
    await page.fill('input[name="username"]', TEST_USERS.admin.username);
    await page.fill('input[name="password"]', TEST_USERS.admin.password);

    // Submit form
    await page.click('button[type="submit"]');

    // Should redirect to dashboard
    await page.waitForURL('/admin/dashboard', { timeout: 10000 });
    await expect(page).toHaveURL(/.*dashboard/);

    // Should show user info
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should reject invalid credentials', async ({ page }) => {
    await page.goto('/login');

    // Fill with invalid credentials
    await page.fill('input[name="username"]', 'invalid@test.com');
    await page.fill('input[name="password"]', 'wrongpassword');

    // Submit form
    await page.click('button[type="submit"]');

    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid credentials');

    // Should remain on login page
    await expect(page).toHaveURL(/.*login/);
  });

  test('should logout successfully', async ({ page }) => {
    // Login first
    await login(page, 'admin');

    // Logout
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');

    // Should redirect to login
    await page.waitForURL('/login', { timeout: 10000 });
    await expect(page).toHaveURL(/.*login/);

    // Token should be cleared from localStorage
    const token = await page.evaluate(() => localStorage.getItem('auth_token'));
    expect(token).toBeNull();
  });

  test('should redirect to login when accessing protected route without auth', async ({ page }) => {
    // Try to access admin dashboard without login
    await page.goto('/admin/dashboard');

    // Should redirect to login
    await page.waitForURL('/login', { timeout: 10000 });
    await expect(page).toHaveURL(/.*login/);
  });

  test('should reject expired token', async ({ page, context }) => {
    // Set expired token in localStorage
    await context.addCookies([]);
    await page.goto('/login');

    // Set expired token
    await page.evaluate(() => {
      localStorage.setItem('auth_token', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2MDAwMDAwMDB9.invalid');
    });

    // Try to access protected route
    await page.goto('/admin/dashboard');

    // Should redirect to login due to expired token
    await page.waitForURL('/login', { timeout: 10000 });
    await expect(page).toHaveURL(/.*login/);
  });

  test('should maintain session across page reloads', async ({ page }) => {
    // Login
    await login(page, 'admin');

    // Reload page
    await page.reload();

    // Should still be authenticated
    await expect(page).toHaveURL(/.*dashboard/);
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should handle 401 responses by redirecting to login', async ({ page }) => {
    // Login first
    await login(page, 'admin');

    // Mock 401 response
    await page.route('**/admin/agents', async (route) => {
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Unauthorized' })
      });
    });

    // Navigate to agents page
    await page.goto('/admin/agents');

    // Should redirect to login
    await page.waitForURL('/login', { timeout: 10000 });
    await expect(page).toHaveURL(/.*login/);
  });

  test('should show loading spinner during login', async ({ page }) => {
    await page.goto('/login');

    // Fill form
    await page.fill('input[name="username"]', TEST_USERS.admin.username);
    await page.fill('input[name="password"]', TEST_USERS.admin.password);

    // Delay response to see loading state
    await page.route('**/auth/login', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.continue();
    });

    // Submit form
    await page.click('button[type="submit"]');

    // Loading spinner should be visible
    await expect(page.locator('[data-testid="loading-spinner"]')).toBeVisible();
  });
});

test.describe('RBAC Authorization', () => {
  test('should allow admin access to all routes', async ({ page }) => {
    await login(page, 'admin');

    // Navigate to admin-only pages
    await page.goto('/admin/agents');
    await expect(page).toHaveURL(/.*agents/);

    await page.goto('/admin/api-keys');
    await expect(page).toHaveURL(/.*api-keys/);

    await page.goto('/admin/jwt');
    await expect(page).toHaveURL(/.*jwt/);
  });

  test('should deny viewer access to admin routes', async ({ page }) => {
    await login(page, 'viewer');

    // Try to access admin route
    await page.goto('/admin/api-keys');

    // Should show 403 error or redirect
    await expect(page.locator('[data-testid="error-403"]')).toBeVisible();
  });

  test('should deny developer access to JWT management', async ({ page }) => {
    await login(page, 'developer');

    // Try to access JWT management (admin-only)
    await page.goto('/admin/jwt');

    // Should show 403 error
    await expect(page.locator('[data-testid="error-403"]')).toBeVisible();
  });
});
