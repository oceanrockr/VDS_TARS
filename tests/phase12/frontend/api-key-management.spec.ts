/**
 * API Key Management E2E tests
 *
 * Tests:
 * - Create API key
 * - Rotate key
 * - Revoke key
 * - One-time key display modal
 * - Copy-to-clipboard functionality
 */

import { test, expect } from '@playwright/test';
import { login, mockAPIResponse, waitForAPICall } from './fixtures';

test.describe('API Key Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await login(page, 'admin');
  });

  test('should display list of API keys', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', {
      keys: [
        {
          id: 'key-1',
          service_name: 'Test Service',
          created_at: '2025-11-15T12:00:00Z',
          expires_at: null,
          is_active: true
        },
        {
          id: 'key-2',
          service_name: 'Another Service',
          created_at: '2025-11-14T10:00:00Z',
          expires_at: '2026-11-14T10:00:00Z',
          is_active: true
        }
      ],
      total: 2
    });

    await page.goto('/admin/api-keys');

    // Should display API key table
    await expect(page.locator('[data-testid="api-key-table"]')).toBeVisible();

    // Should display service names
    await expect(page.locator('text=Test Service')).toBeVisible();
    await expect(page.locator('text=Another Service')).toBeVisible();

    // Should display active badges
    await expect(page.locator('[data-testid="badge-active"]')).toHaveCount(2);
  });

  test('should create new API key successfully', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', { keys: [], total: 0 });

    await mockAPIResponse(page, '/api-keys', {
      id: 'new-key-id',
      key: 'tars_1234567890abcdef',
      service_name: 'New Service',
      created_at: '2025-11-15T12:00:00Z',
      expires_at: null,
      message: 'WARNING: Copy this key now. It will not be shown again.'
    }, 'POST');

    await page.goto('/admin/api-keys');

    // Click create button
    await page.click('[data-testid="create-api-key-button"]');

    // Fill form
    await page.fill('[data-testid="service-name-input"]', 'New Service');

    // Submit form
    await page.click('[data-testid="submit-create-key"]');

    // Wait for API call
    await waitForAPICall(page, '/api-keys', 'POST');

    // Should show new key modal
    await expect(page.locator('[data-testid="new-key-modal"]')).toBeVisible();
    await expect(page.locator('[data-testid="new-key-display"]')).toContainText('tars_1234567890abcdef');

    // Should show warning message
    await expect(page.locator('text=Copy this key now')).toBeVisible();
  });

  test('should create API key with expiration', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', { keys: [], total: 0 });

    await mockAPIResponse(page, '/api-keys', {
      id: 'new-key-id',
      key: 'tars_abcdef123456',
      service_name: 'Expiring Service',
      created_at: '2025-11-15T12:00:00Z',
      expires_at: '2026-11-15T12:00:00Z'
    }, 'POST');

    await page.goto('/admin/api-keys');

    // Click create button
    await page.click('[data-testid="create-api-key-button"]');

    // Fill form with expiration
    await page.fill('[data-testid="service-name-input"]', 'Expiring Service');
    await page.fill('[data-testid="expires-in-days-input"]', '365');

    // Submit form
    await page.click('[data-testid="submit-create-key"]');

    // Wait for API call
    await waitForAPICall(page, '/api-keys', 'POST');

    // Should show expiration date in modal
    await expect(page.locator('[data-testid="new-key-modal"]')).toContainText('2026-11-15');
  });

  test('should copy new key to clipboard', async ({ page, context }) => {
    // Grant clipboard permissions
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);

    await mockAPIResponse(page, '/api-keys', { keys: [], total: 0 });

    await mockAPIResponse(page, '/api-keys', {
      id: 'new-key-id',
      key: 'tars_1234567890abcdef',
      service_name: 'New Service',
      created_at: '2025-11-15T12:00:00Z'
    }, 'POST');

    await page.goto('/admin/api-keys');

    // Create key
    await page.click('[data-testid="create-api-key-button"]');
    await page.fill('[data-testid="service-name-input"]', 'New Service');
    await page.click('[data-testid="submit-create-key"]');

    // Wait for modal
    await page.waitForSelector('[data-testid="new-key-modal"]');

    // Click copy button
    await page.click('[data-testid="copy-key-button"]');

    // Verify clipboard content
    const clipboardText = await page.evaluate(() => navigator.clipboard.readText());
    expect(clipboardText).toBe('tars_1234567890abcdef');

    // Should show "Copied!" feedback
    await expect(page.locator('text=Copied!')).toBeVisible();
  });

  test('should validate create key form', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', { keys: [], total: 0 });

    await page.goto('/admin/api-keys');

    // Click create button
    await page.click('[data-testid="create-api-key-button"]');

    // Try to submit without service name
    await page.click('[data-testid="submit-create-key"]');

    // Should show validation error
    await expect(page.locator('[data-testid="validation-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="validation-error"]')).toContainText('Service name is required');
  });

  test('should rotate API key successfully', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', {
      keys: [
        {
          id: 'key-1',
          service_name: 'Test Service',
          created_at: '2025-11-15T12:00:00Z',
          is_active: true
        }
      ],
      total: 1
    });

    await mockAPIResponse(page, '/api-keys/key-1/rotate', {
      id: 'key-1',
      key: 'tars_newkey123456',
      service_name: 'Test Service',
      rotated_at: '2025-11-15T13:00:00Z',
      message: 'Key rotated successfully. Old key is now invalid.'
    }, 'POST');

    await page.goto('/admin/api-keys');

    // Click rotate button
    await page.click('[data-testid="rotate-button-key-1"]');

    // Confirm rotation
    await page.click('[data-testid="confirm-rotate"]');

    // Wait for API call
    await waitForAPICall(page, '/api-keys/key-1/rotate', 'POST');

    // Should show new key modal
    await expect(page.locator('[data-testid="new-key-modal"]')).toBeVisible();
    await expect(page.locator('[data-testid="new-key-display"]')).toContainText('tars_newkey123456');
  });

  test('should revoke API key with reason', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', {
      keys: [
        {
          id: 'key-1',
          service_name: 'Test Service',
          created_at: '2025-11-15T12:00:00Z',
          is_active: true
        }
      ],
      total: 1
    });

    await mockAPIResponse(page, '/api-keys/key-1/revoke', {
      success: true,
      message: 'API key revoked successfully'
    }, 'POST');

    await page.goto('/admin/api-keys');

    // Click revoke button
    await page.click('[data-testid="revoke-button-key-1"]');

    // Fill revocation reason
    await page.fill('[data-testid="revocation-reason-input"]', 'Security incident');

    // Confirm revocation
    await page.click('[data-testid="confirm-revoke"]');

    // Wait for API call
    await waitForAPICall(page, '/api-keys/key-1/revoke', 'POST');

    // Should show success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="success-message"]')).toContainText('revoked successfully');
  });

  test('should display revoked keys with badge', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', {
      keys: [
        {
          id: 'key-1',
          service_name: 'Active Service',
          is_active: true
        },
        {
          id: 'key-2',
          service_name: 'Revoked Service',
          is_active: false,
          revoked_at: '2025-11-14T10:00:00Z'
        }
      ],
      total: 2
    });

    await page.goto('/admin/api-keys');

    // Should display different badges
    await expect(page.locator('[data-testid="badge-active"]')).toHaveCount(1);
    await expect(page.locator('[data-testid="badge-revoked"]')).toHaveCount(1);
  });

  test('should disable rotate/revoke for inactive keys', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', {
      keys: [
        {
          id: 'key-revoked',
          service_name: 'Revoked Service',
          is_active: false
        }
      ],
      total: 1
    });

    await page.goto('/admin/api-keys');

    // Rotate and revoke buttons should be disabled
    await expect(page.locator('[data-testid="rotate-button-key-revoked"]')).toBeDisabled();
    await expect(page.locator('[data-testid="revoke-button-key-revoked"]')).toBeDisabled();
  });

  test('should handle create key error gracefully', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', { keys: [], total: 0 });

    // Mock error response
    await page.route('**/admin/api-keys', async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'Failed to create API key' })
        });
      } else {
        await route.continue();
      }
    });

    await page.goto('/admin/api-keys');

    // Try to create key
    await page.click('[data-testid="create-api-key-button"]');
    await page.fill('[data-testid="service-name-input"]', 'Test Service');
    await page.click('[data-testid="submit-create-key"]');

    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Failed to create API key');
  });

  test('should display empty state when no API keys', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', { keys: [], total: 0 });

    await page.goto('/admin/api-keys');

    // Should show empty state
    await expect(page.locator('[data-testid="empty-state"]')).toBeVisible();
    await expect(page.locator('text=No API keys found')).toBeVisible();
  });

  test('should display loading state while fetching keys', async ({ page }) => {
    // Delay response
    await page.route('**/admin/api-keys', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.continue();
    });

    await page.goto('/admin/api-keys');

    // Should show loading spinner
    await expect(page.locator('[data-testid="loading-spinner"]')).toBeVisible();
  });

  test('should sort API keys by created date', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', {
      keys: [
        {
          id: 'key-1',
          service_name: 'Newer Service',
          created_at: '2025-11-15T12:00:00Z'
        },
        {
          id: 'key-2',
          service_name: 'Older Service',
          created_at: '2025-11-14T12:00:00Z'
        }
      ],
      total: 2
    });

    await page.goto('/admin/api-keys');

    // Click sort header
    await page.click('[data-testid="sort-created-at"]');

    // First row should be older service (ascending order)
    const firstRow = page.locator('[data-testid="api-key-row"]').first();
    await expect(firstRow).toContainText('Older Service');
  });

  test('should close new key modal after copy', async ({ page }) => {
    await mockAPIResponse(page, '/api-keys', { keys: [], total: 0 });

    await mockAPIResponse(page, '/api-keys', {
      id: 'new-key-id',
      key: 'tars_1234567890abcdef',
      service_name: 'New Service'
    }, 'POST');

    await page.goto('/admin/api-keys');

    // Create key
    await page.click('[data-testid="create-api-key-button"]');
    await page.fill('[data-testid="service-name-input"]', 'New Service');
    await page.click('[data-testid="submit-create-key"]');

    // Wait for modal
    await page.waitForSelector('[data-testid="new-key-modal"]');

    // Close modal
    await page.click('[data-testid="close-modal-button"]');

    // Modal should be closed
    await expect(page.locator('[data-testid="new-key-modal"]')).not.toBeVisible();
  });
});
