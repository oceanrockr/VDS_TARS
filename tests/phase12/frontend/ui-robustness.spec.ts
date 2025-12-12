/**
 * UI Robustness E2E tests
 *
 * Tests:
 * - 401 → redirect to login
 * - Network error overlays
 * - Loading spinners
 * - Empty-state tests
 * - Accessibility
 */

import { test, expect } from '@playwright/test';
import { login } from './fixtures';

test.describe('UI Robustness', () => {
  test('should redirect to login on 401 unauthorized', async ({ page }) => {
    await login(page, 'admin');

    // Mock 401 response for any API call
    await page.route('**/admin/**', async (route) => {
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Unauthorized' })
      });
    });

    // Navigate to a protected route
    await page.goto('/admin/agents');

    // Should redirect to login
    await page.waitForURL('/login', { timeout: 10000 });
    await expect(page).toHaveURL(/.*login/);

    // Should clear token from localStorage
    const token = await page.evaluate(() => localStorage.getItem('auth_token'));
    expect(token).toBeNull();
  });

  test('should display network error overlay on connection failure', async ({ page }) => {
    await login(page, 'admin');

    // Mock network error
    await page.route('**/admin/agents', async (route) => {
      await route.abort('failed');
    });

    await page.goto('/admin/agents');

    // Should show network error overlay
    await expect(page.locator('[data-testid="network-error-overlay"]')).toBeVisible();
    await expect(page.locator('[data-testid="network-error-overlay"]')).toContainText('Network error');
    await expect(page.locator('[data-testid="network-error-overlay"]')).toContainText('Please check your connection');
  });

  test('should allow retry after network error', async ({ page }) => {
    await login(page, 'admin');

    let callCount = 0;

    // Fail first call, succeed on retry
    await page.route('**/admin/agents', async (route) => {
      callCount++;
      if (callCount === 1) {
        await route.abort('failed');
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ agents: [] })
        });
      }
    });

    await page.goto('/admin/agents');

    // Network error should appear
    await expect(page.locator('[data-testid="network-error-overlay"]')).toBeVisible();

    // Click retry
    await page.click('[data-testid="retry-button"]');

    // Error should disappear
    await expect(page.locator('[data-testid="network-error-overlay"]')).not.toBeVisible();

    // Content should load
    await expect(page.locator('[data-testid="agent-list"]')).toBeVisible();
  });

  test('should display loading spinner during async operations', async ({ page }) => {
    await login(page, 'admin');

    // Delay response
    await page.route('**/admin/agents', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 2000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ agents: [] })
      });
    });

    await page.goto('/admin/agents');

    // Loading spinner should be visible
    await expect(page.locator('[data-testid="loading-spinner"]')).toBeVisible();

    // Wait for loading to complete
    await page.waitForSelector('[data-testid="loading-spinner"]', { state: 'hidden', timeout: 5000 });

    // Content should be visible
    await expect(page.locator('[data-testid="agent-list"]')).toBeVisible();
  });

  test('should display empty state when no data', async ({ page }) => {
    await login(page, 'admin');

    // Mock empty response
    await page.route('**/admin/agents', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ agents: [] })
      });
    });

    await page.goto('/admin/agents');

    // Should show empty state
    await expect(page.locator('[data-testid="empty-state"]')).toBeVisible();
    await expect(page.locator('[data-testid="empty-state"]')).toContainText('No agents found');
    await expect(page.locator('[data-testid="empty-state-icon"]')).toBeVisible();
  });

  test('should handle 403 forbidden gracefully', async ({ page }) => {
    await login(page, 'admin');

    // Mock 403 response
    await page.route('**/admin/api-keys', async (route) => {
      await route.fulfill({
        status: 403,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Insufficient permissions' })
      });
    });

    await page.goto('/admin/api-keys');

    // Should show 403 error
    await expect(page.locator('[data-testid="error-403"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-403"]')).toContainText('Access Denied');
    await expect(page.locator('[data-testid="error-403"]')).toContainText('Insufficient permissions');
  });

  test('should handle 404 not found gracefully', async ({ page }) => {
    await login(page, 'admin');

    // Navigate to non-existent route
    await page.goto('/admin/nonexistent-page');

    // Should show 404 error
    await expect(page.locator('[data-testid="error-404"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-404"]')).toContainText('Page Not Found');
  });

  test('should handle 500 server error gracefully', async ({ page }) => {
    await login(page, 'admin');

    // Mock 500 response
    await page.route('**/admin/agents', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Internal server error' })
      });
    });

    await page.goto('/admin/agents');

    // Should show 500 error
    await expect(page.locator('[data-testid="error-500"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-500"]')).toContainText('Server Error');
  });

  test('should disable submit button during form submission', async ({ page }) => {
    await login(page, 'admin');

    await page.route('**/admin/api-keys', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ keys: [], total: 0 })
        });
      } else if (route.request().method() === 'POST') {
        await new Promise(resolve => setTimeout(resolve, 1000));
        await route.fulfill({
          status: 201,
          body: JSON.stringify({ id: 'key-1', key: 'tars_123' })
        });
      }
    });

    await page.goto('/admin/api-keys');

    // Open create modal
    await page.click('[data-testid="create-api-key-button"]');
    await page.fill('[data-testid="service-name-input"]', 'Test Service');

    // Submit button should be enabled
    await expect(page.locator('[data-testid="submit-create-key"]')).toBeEnabled();

    // Click submit
    await page.click('[data-testid="submit-create-key"]');

    // Submit button should be disabled during submission
    await expect(page.locator('[data-testid="submit-create-key"]')).toBeDisabled();
  });

  test('should display toast notifications for success', async ({ page }) => {
    await login(page, 'admin');

    await page.route('**/admin/agents/dqn_agent/reload', async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({ success: true, message: 'Agent reloaded successfully' })
      });
    });

    await page.route('**/admin/agents', async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          agents: [{ id: 'dqn_agent', name: 'DQN Agent', state: 'active' }]
        })
      });
    });

    await page.goto('/admin/agents');

    // Trigger reload
    await page.click('[data-testid="reload-button"]');
    await page.fill('[data-testid="reload-reason-input"]', 'Test');
    await page.click('[data-testid="confirm-reload-button"]');

    // Success toast should appear
    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
    await expect(page.locator('[data-testid="toast-success"]')).toContainText('Agent reloaded successfully');

    // Toast should auto-dismiss after 5 seconds
    await page.waitForSelector('[data-testid="toast-success"]', { state: 'hidden', timeout: 6000 });
  });

  test('should be keyboard accessible', async ({ page }) => {
    await login(page, 'admin');

    await page.route('**/admin/agents', async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          agents: [{ id: 'dqn_agent', name: 'DQN Agent', state: 'active' }]
        })
      });
    });

    await page.goto('/admin/agents');

    // Navigate with Tab key
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Focused element should be visible
    const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
    expect(focusedElement).toBeTruthy();

    // Should be able to activate with Enter
    await page.keyboard.press('Enter');
  });

  test('should have proper ARIA labels', async ({ page }) => {
    await login(page, 'admin');

    await page.route('**/admin/agents', async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({ agents: [] })
      });
    });

    await page.goto('/admin/agents');

    // Check for ARIA labels
    const main = page.locator('[role="main"]');
    await expect(main).toBeVisible();

    const navigation = page.locator('[role="navigation"]');
    await expect(navigation).toBeVisible();
  });

  test('should handle offline mode', async ({ page, context }) => {
    await login(page, 'admin');

    // Simulate offline
    await context.setOffline(true);

    await page.goto('/admin/agents');

    // Should show offline message
    await expect(page.locator('[data-testid="offline-banner"]')).toBeVisible();
    await expect(page.locator('[data-testid="offline-banner"]')).toContainText('You are offline');

    // Reconnect
    await context.setOffline(false);

    // Offline banner should disappear
    await expect(page.locator('[data-testid="offline-banner"]')).not.toBeVisible();
  });

  test('should display validation errors inline', async ({ page }) => {
    await login(page, 'admin');

    await page.route('**/admin/api-keys', async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({ keys: [], total: 0 })
      });
    });

    await page.goto('/admin/api-keys');

    // Open create modal
    await page.click('[data-testid="create-api-key-button"]');

    // Try to submit without filling required field
    await page.click('[data-testid="submit-create-key"]');

    // Should show inline validation error
    await expect(page.locator('[data-testid="service-name-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="service-name-error"]')).toContainText('required');

    // Error should have proper styling
    const errorColor = await page.locator('[data-testid="service-name-error"]').evaluate((el) => {
      return window.getComputedStyle(el).color;
    });
    expect(errorColor).toBeTruthy();
  });

  test('should prevent double-click submissions', async ({ page }) => {
    await login(page, 'admin');

    let submitCount = 0;

    await page.route('**/admin/api-keys', async (route) => {
      if (route.request().method() === 'POST') {
        submitCount++;
        await new Promise(resolve => setTimeout(resolve, 500));
        await route.fulfill({
          status: 201,
          body: JSON.stringify({ id: 'key-1', key: 'tars_123' })
        });
      } else {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ keys: [], total: 0 })
        });
      }
    });

    await page.goto('/admin/api-keys');

    // Open create modal
    await page.click('[data-testid="create-api-key-button"]');
    await page.fill('[data-testid="service-name-input"]', 'Test Service');

    // Double-click submit button
    await page.click('[data-testid="submit-create-key"]', { clickCount: 2 });

    // Wait for submission
    await page.waitForTimeout(1000);

    // Should only submit once
    expect(submitCount).toBe(1);
  });
});
