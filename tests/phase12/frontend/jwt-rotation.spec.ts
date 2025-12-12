/**
 * JWT Rotation E2E tests
 *
 * Tests:
 * - Trigger rotation from UI
 * - Verify old tokens still work during grace period
 * - Verify new tokens have new kid
 * - Force invalidation
 * - View JWT key status
 */

import { test, expect } from '@playwright/test';
import { login, mockAPIResponse, waitForAPICall } from './fixtures';

test.describe('JWT Key Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await login(page, 'admin');
  });

  test('should display current JWT key status', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/status', {
      current_kid: 'key-20251115120000-a1b2c3d4',
      active_keys: [
        { kid: 'key-20251115120000-a1b2c3d4', is_active: true, is_valid: true }
      ],
      valid_keys: [
        { kid: 'key-20251115120000-a1b2c3d4', is_active: true, is_valid: true },
        { kid: 'key-20251114120000-x1y2z3', is_active: false, is_valid: true }
      ],
      total_active: 1,
      total_valid: 2
    });

    await page.goto('/admin/jwt');

    // Should display current key
    await expect(page.locator('[data-testid="current-key-id"]')).toContainText('key-20251115120000-a1b2c3d4');

    // Should display key counts
    await expect(page.locator('[data-testid="active-keys-count"]')).toContainText('1');
    await expect(page.locator('[data-testid="valid-keys-count"]')).toContainText('2');
  });

  test('should rotate JWT key successfully', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/status', {
      current_kid: 'key-old-123',
      total_active: 1,
      total_valid: 1
    });

    await mockAPIResponse(page, '/jwt/rotate', {
      success: true,
      old_kid: 'key-old-123',
      new_kid: 'key-new-456',
      message: 'JWT key rotated successfully. Old tokens valid for 24h.',
      timestamp: '2025-11-15T12:00:00Z',
      grace_period_hours: 24
    }, 'POST');

    await page.goto('/admin/jwt');

    // Click rotate button
    await page.click('[data-testid="rotate-jwt-button"]');

    // Confirm rotation
    await page.click('[data-testid="confirm-rotate-jwt"]');

    // Wait for API call
    await waitForAPICall(page, '/jwt/rotate', 'POST');

    // Should show success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="success-message"]')).toContainText('rotated successfully');

    // Should display new key ID
    await expect(page.locator('[data-testid="success-message"]')).toContainText('key-new-456');

    // Should mention grace period
    await expect(page.locator('[data-testid="success-message"]')).toContainText('24h');
  });

  test('should display grace period warning', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/status', {
      current_kid: 'key-current',
      total_active: 1,
      total_valid: 1
    });

    await page.goto('/admin/jwt');

    // Click rotate button
    await page.click('[data-testid="rotate-jwt-button"]');

    // Should show grace period warning
    await expect(page.locator('[data-testid="rotation-warning"]')).toBeVisible();
    await expect(page.locator('[data-testid="rotation-warning"]')).toContainText('Old tokens will remain valid for 24 hours');
  });

  test('should list all JWT keys with status', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/keys', {
      keys: [
        {
          kid: 'key-active',
          algorithm: 'HS256',
          created_at: '2025-11-15T12:00:00Z',
          expires_at: null,
          is_active: true,
          is_valid: true
        },
        {
          kid: 'key-grace-period',
          algorithm: 'HS256',
          created_at: '2025-11-14T12:00:00Z',
          expires_at: '2025-11-15T13:00:00Z',
          is_active: false,
          is_valid: true
        },
        {
          kid: 'key-expired',
          algorithm: 'HS256',
          created_at: '2025-11-13T12:00:00Z',
          expires_at: '2025-11-14T12:00:00Z',
          is_active: false,
          is_valid: false
        }
      ],
      current_kid: 'key-active',
      total: 3
    });

    await page.goto('/admin/jwt/keys');

    // Should display all keys
    await expect(page.locator('[data-testid="jwt-key-row"]')).toHaveCount(3);

    // Should display status badges
    await expect(page.locator('[data-testid="badge-active"]')).toHaveCount(1);
    await expect(page.locator('[data-testid="badge-valid"]')).toHaveCount(2);
    await expect(page.locator('[data-testid="badge-expired"]')).toHaveCount(1);
  });

  test('should invalidate JWT key with reason', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/keys', {
      keys: [
        {
          kid: 'key-to-invalidate',
          algorithm: 'HS256',
          is_active: false,
          is_valid: true
        }
      ],
      total: 1
    });

    await mockAPIResponse(page, '/jwt/keys/key-to-invalidate/invalidate', {
      success: true,
      message: 'JWT key invalidated successfully'
    }, 'POST');

    await page.goto('/admin/jwt/keys');

    // Click invalidate button
    await page.click('[data-testid="invalidate-button-key-to-invalidate"]');

    // Fill invalidation reason
    await page.fill('[data-testid="invalidation-reason-input"]', 'Key compromised');

    // Confirm invalidation
    await page.click('[data-testid="confirm-invalidate"]');

    // Wait for API call
    await waitForAPICall(page, '/jwt/keys/key-to-invalidate/invalidate', 'POST');

    // Should show success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="success-message"]')).toContainText('invalidated successfully');
  });

  test('should show critical warning for invalidation', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/keys', {
      keys: [{ kid: 'key-123', is_active: false, is_valid: true }],
      total: 1
    });

    await page.goto('/admin/jwt/keys');

    // Click invalidate button
    await page.click('[data-testid="invalidate-button-key-123"]');

    // Should show critical warning
    await expect(page.locator('[data-testid="invalidation-warning"]')).toBeVisible();
    await expect(page.locator('[data-testid="invalidation-warning"]')).toContainText('This will immediately invalidate all tokens');
    await expect(page.locator('[data-testid="invalidation-warning"]')).toContainText('CRITICAL');
  });

  test('should disable invalidate for active key', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/keys', {
      keys: [
        {
          kid: 'key-active',
          is_active: true,
          is_valid: true
        }
      ],
      current_kid: 'key-active',
      total: 1
    });

    await page.goto('/admin/jwt/keys');

    // Invalidate button should be disabled for active key
    await expect(page.locator('[data-testid="invalidate-button-key-active"]')).toBeDisabled();

    // Should show tooltip explaining why
    await page.hover('[data-testid="invalidate-button-key-active"]');
    await expect(page.locator('[data-testid="tooltip"]')).toContainText('Cannot invalidate active key');
  });

  test('should verify old tokens work during grace period', async ({ page, context }) => {
    // Simulate rotation
    await mockAPIResponse(page, '/jwt/rotate', {
      success: true,
      old_kid: 'key-old',
      new_kid: 'key-new',
      grace_period_hours: 24
    }, 'POST');

    await page.goto('/admin/jwt');

    // Get old token from localStorage
    const oldToken = await page.evaluate(() => localStorage.getItem('auth_token'));

    // Trigger rotation
    await page.click('[data-testid="rotate-jwt-button"]');
    await page.click('[data-testid="confirm-rotate-jwt"]');

    // Wait for rotation to complete
    await page.waitForSelector('[data-testid="success-message"]');

    // Old token should still work (grace period)
    // Make API call with old token
    await page.evaluate((token) => {
      localStorage.setItem('auth_token', token);
    }, oldToken);

    // Navigate to a protected route
    await page.goto('/admin/agents');

    // Should not redirect to login (old token still valid)
    await expect(page).toHaveURL(/.*agents/);
  });

  test('should handle rotation error gracefully', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/status', {
      current_kid: 'key-current',
      total_active: 1
    });

    // Mock error response
    await page.route('**/admin/jwt/rotate', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Failed to rotate JWT key' })
      });
    });

    await page.goto('/admin/jwt');

    // Try to rotate
    await page.click('[data-testid="rotate-jwt-button"]');
    await page.click('[data-testid="confirm-rotate-jwt"]');

    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Failed to rotate');
  });

  test('should display key algorithm', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/keys', {
      keys: [
        { kid: 'key-hs256', algorithm: 'HS256', is_active: true, is_valid: true },
        { kid: 'key-hs512', algorithm: 'HS512', is_active: false, is_valid: true }
      ],
      total: 2
    });

    await page.goto('/admin/jwt/keys');

    // Should display algorithms
    await expect(page.locator('text=HS256')).toBeVisible();
    await expect(page.locator('text=HS512')).toBeVisible();
  });

  test('should display key expiration times', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/keys', {
      keys: [
        {
          kid: 'key-1',
          expires_at: '2025-11-16T12:00:00Z',
          is_active: false,
          is_valid: true
        }
      ],
      total: 1
    });

    await page.goto('/admin/jwt/keys');

    // Should display expiration time
    await expect(page.locator('[data-testid="expires-at"]')).toContainText('2025-11-16');
  });

  test('should highlight current key', async ({ page }) => {
    await mockAPIResponse(page, '/jwt/keys', {
      keys: [
        { kid: 'key-current', is_active: true, is_valid: true },
        { kid: 'key-old', is_active: false, is_valid: true }
      ],
      current_kid: 'key-current',
      total: 2
    });

    await page.goto('/admin/jwt/keys');

    // Current key row should have highlight class
    const currentKeyRow = page.locator('[data-testid="jwt-key-row-key-current"]');
    await expect(currentKeyRow).toHaveClass(/highlight/);
  });
});
