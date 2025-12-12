/**
 * Agent Management E2E tests
 *
 * Tests:
 * - View agents list
 * - Reload agent flow
 * - Promote model flow
 * - Error handling for failed reload
 * - Auto-refresh functionality
 */

import { test, expect } from '@playwright/test';
import { login, mockAPIResponse, waitForAPICall } from './fixtures';

test.describe('Agent Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await login(page, 'admin');
  });

  test('should display list of agents', async ({ page }) => {
    // Mock agents response
    await mockAPIResponse(page, '/agents', {
      agents: [
        {
          id: 'dqn_agent',
          name: 'DQN Agent',
          state: 'active',
          algorithm: 'DQN',
          reward: 0.75,
          loss: 0.23,
          entropy: 0.45
        },
        {
          id: 'ppo_agent',
          name: 'PPO Agent',
          state: 'training',
          algorithm: 'PPO',
          reward: 0.82,
          loss: 0.18,
          entropy: 0.52
        }
      ]
    });

    // Navigate to agents page
    await page.goto('/admin/agents');

    // Should display agent cards
    await expect(page.locator('[data-testid="agent-card"]')).toHaveCount(2);

    // Should display agent names
    await expect(page.locator('text=DQN Agent')).toBeVisible();
    await expect(page.locator('text=PPO Agent')).toBeVisible();

    // Should display performance metrics
    await expect(page.locator('text=0.75')).toBeVisible(); // DQN reward
    await expect(page.locator('text=0.82')).toBeVisible(); // PPO reward
  });

  test('should display agent state badges', async ({ page }) => {
    await mockAPIResponse(page, '/agents', {
      agents: [
        { id: 'agent1', name: 'Active Agent', state: 'active', algorithm: 'DQN', reward: 0.75 },
        { id: 'agent2', name: 'Training Agent', state: 'training', algorithm: 'PPO', reward: 0.82 },
        { id: 'agent3', name: 'Inactive Agent', state: 'inactive', algorithm: 'A2C', reward: 0.60 }
      ]
    });

    await page.goto('/admin/agents');

    // Should display state badges with correct colors
    await expect(page.locator('[data-testid="badge-active"]')).toBeVisible();
    await expect(page.locator('[data-testid="badge-training"]')).toBeVisible();
    await expect(page.locator('[data-testid="badge-inactive"]')).toBeVisible();
  });

  test('should reload agent successfully', async ({ page }) => {
    await mockAPIResponse(page, '/agents', {
      agents: [
        { id: 'dqn_agent', name: 'DQN Agent', state: 'active', algorithm: 'DQN', reward: 0.75 }
      ]
    });

    await mockAPIResponse(page, '/agents/dqn_agent/reload', {
      success: true,
      message: 'Agent reloaded successfully'
    }, 'POST');

    await page.goto('/admin/agents');

    // Find agent card and click reload
    const agentCard = page.locator('[data-testid="agent-card"]').first();
    await agentCard.locator('[data-testid="reload-button"]').click();

    // Fill reload reason
    await page.fill('[data-testid="reload-reason-input"]', 'Performance degradation');
    await page.click('[data-testid="confirm-reload-button"]');

    // Wait for API call
    await waitForAPICall(page, '/agents/dqn_agent/reload', 'POST');

    // Should show success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="success-message"]')).toContainText('Agent reloaded successfully');
  });

  test('should validate reload reason input', async ({ page }) => {
    await mockAPIResponse(page, '/agents', {
      agents: [
        { id: 'dqn_agent', name: 'DQN Agent', state: 'active', algorithm: 'DQN', reward: 0.75 }
      ]
    });

    await page.goto('/admin/agents');

    // Click reload without entering reason
    const agentCard = page.locator('[data-testid="agent-card"]').first();
    await agentCard.locator('[data-testid="reload-button"]').click();

    // Try to confirm without reason
    await page.click('[data-testid="confirm-reload-button"]');

    // Should show validation error
    await expect(page.locator('[data-testid="validation-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="validation-error"]')).toContainText('Reason is required');
  });

  test('should promote model successfully', async ({ page }) => {
    await mockAPIResponse(page, '/agents', {
      agents: [
        { id: 'dqn_agent', name: 'DQN Agent', state: 'active', algorithm: 'DQN', reward: 0.75 }
      ]
    });

    await mockAPIResponse(page, '/agents/dqn_agent/promote', {
      success: true,
      message: 'Model promoted to production'
    }, 'POST');

    await page.goto('/admin/agents');

    // Find agent card and click promote
    const agentCard = page.locator('[data-testid="agent-card"]').first();
    await agentCard.locator('[data-testid="promote-button"]').click();

    // Fill promotion details
    await page.fill('[data-testid="model-version-input"]', 'v1.2.0');
    await page.fill('[data-testid="promotion-reason-input"]', 'Reward improvement +15%');
    await page.click('[data-testid="confirm-promote-button"]');

    // Wait for API call
    await waitForAPICall(page, '/agents/dqn_agent/promote', 'POST');

    // Should show success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="success-message"]')).toContainText('Model promoted');
  });

  test('should display hyperparameters in collapsible section', async ({ page }) => {
    await mockAPIResponse(page, '/agents/dqn_agent', {
      id: 'dqn_agent',
      name: 'DQN Agent',
      state: 'active',
      algorithm: 'DQN',
      reward: 0.75,
      hyperparameters: {
        learning_rate: 0.001,
        gamma: 0.99,
        epsilon: 0.1
      }
    });

    await mockAPIResponse(page, '/agents', {
      agents: [
        { id: 'dqn_agent', name: 'DQN Agent', state: 'active', algorithm: 'DQN', reward: 0.75 }
      ]
    });

    await page.goto('/admin/agents');

    // Click to expand hyperparameters
    await page.click('[data-testid="expand-hyperparameters"]');

    // Should display hyperparameters
    await expect(page.locator('text=learning_rate')).toBeVisible();
    await expect(page.locator('text=0.001')).toBeVisible();
    await expect(page.locator('text=gamma')).toBeVisible();
    await expect(page.locator('text=0.99')).toBeVisible();
  });

  test('should handle reload error gracefully', async ({ page }) => {
    await mockAPIResponse(page, '/agents', {
      agents: [
        { id: 'dqn_agent', name: 'DQN Agent', state: 'active', algorithm: 'DQN', reward: 0.75 }
      ]
    });

    // Mock error response
    await page.route('**/admin/agents/dqn_agent/reload', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Internal server error' })
      });
    });

    await page.goto('/admin/agents');

    // Try to reload
    const agentCard = page.locator('[data-testid="agent-card"]').first();
    await agentCard.locator('[data-testid="reload-button"]').click();
    await page.fill('[data-testid="reload-reason-input"]', 'Test');
    await page.click('[data-testid="confirm-reload-button"]');

    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Failed to reload agent');
  });

  test('should auto-refresh agent data', async ({ page }) => {
    let requestCount = 0;

    await page.route('**/admin/agents', async (route) => {
      requestCount++;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          agents: [
            { id: 'dqn_agent', name: 'DQN Agent', state: 'active', algorithm: 'DQN', reward: 0.75 + requestCount * 0.01 }
          ]
        })
      });
    });

    await page.goto('/admin/agents');

    // Wait for initial load
    await page.waitForTimeout(1000);
    const initialRequests = requestCount;

    // Wait for auto-refresh (should happen every 30 seconds, but we can speed up for tests)
    await page.waitForTimeout(31000);

    // Should have made additional request
    expect(requestCount).toBeGreaterThan(initialRequests);
  });

  test('should display empty state when no agents', async ({ page }) => {
    await mockAPIResponse(page, '/agents', { agents: [] });

    await page.goto('/admin/agents');

    // Should show empty state
    await expect(page.locator('[data-testid="empty-state"]')).toBeVisible();
    await expect(page.locator('text=No agents found')).toBeVisible();
  });

  test('should display loading state while fetching agents', async ({ page }) => {
    // Delay response
    await page.route('**/admin/agents', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.continue();
    });

    await page.goto('/admin/agents');

    // Should show loading spinner
    await expect(page.locator('[data-testid="loading-spinner"]')).toBeVisible();
  });
});
