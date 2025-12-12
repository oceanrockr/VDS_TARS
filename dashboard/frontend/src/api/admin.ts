/**
 * T.A.R.S. Admin Dashboard API Client
 * Type-safe API client for admin operations
 * Phase 12 Part 2
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

// ============================================================================
// Types
// ============================================================================

export interface Agent {
  agent_id: string;
  agent_type: string;
  state: 'active' | 'inactive' | 'training';
  hyperparameters: Record<string, any>;
  performance_metrics: {
    reward: number;
    loss: number;
    entropy: number;
  };
  model_version: string;
  created_at: string;
  updated_at: string;
}

export interface AutoMLTrial {
  trial_id: string;
  agent_id: string;
  hyperparameters: Record<string, any>;
  score: number;
  status: 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at?: string;
}

export interface HyperSyncProposal {
  proposal_id: string;
  agent_id: string;
  hyperparameters: Record<string, any>;
  status: 'pending' | 'approved' | 'denied';
  created_at: string;
  approved_at?: string;
  approved_by?: string;
}

export interface APIKey {
  key_id: string;
  service_name: string;
  created_at: string;
  last_used?: string;
  is_active: boolean;
}

export interface JWTKey {
  kid: string;
  algorithm: string;
  created_at: string;
  expires_at?: string;
  is_active: boolean;
  is_valid: boolean;
}

export interface SystemHealth {
  status: string;
  services: Record<string, any>;
  timestamp: string;
  overall_healthy: boolean;
}

export interface AuditLogEvent {
  event_id: string;
  event_type: string;
  username: string;
  timestamp: string;
  severity: string;
  metadata: Record<string, any>;
  ip_address: string;
}

// ============================================================================
// Admin API Client
// ============================================================================

export class AdminAPIClient {
  private client: AxiosInstance;

  constructor(token?: string) {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/admin`,
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      timeout: 30000,
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Token expired or invalid
          console.error('Authentication failed');
          // Trigger logout or token refresh
        }
        return Promise.reject(error);
      }
    );
  }

  setToken(token: string) {
    this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  }

  // ============================================================================
  // Agent Management
  // ============================================================================

  async getAllAgents(): Promise<Agent[]> {
    const response = await this.client.get('/agents');
    return response.data.agents || response.data;
  }

  async getAgent(agentId: string): Promise<Agent> {
    const response = await this.client.get(`/agents/${agentId}`);
    return response.data;
  }

  async reloadAgent(agentId: string, reason: string, config?: Record<string, any>) {
    const response = await this.client.post(`/agents/${agentId}/reload`, {
      agent_id: agentId,
      reason,
      config,
    });
    return response.data;
  }

  async promoteModel(agentId: string, modelVersion: string, reason: string) {
    const response = await this.client.post(`/agents/${agentId}/promote`, {
      agent_id: agentId,
      model_version: modelVersion,
      reason,
    });
    return response.data;
  }

  // ============================================================================
  // AutoML Management
  // ============================================================================

  async getAutoMLTrials(agentId?: string, limit: number = 50): Promise<AutoMLTrial[]> {
    const params = { limit, ...(agentId && { agent_id: agentId }) };
    const response = await this.client.get('/automl/trials', { params });
    return response.data.trials || response.data;
  }

  async getAutoMLTrial(trialId: string): Promise<AutoMLTrial> {
    const response = await this.client.get(`/automl/trials/${trialId}`);
    return response.data;
  }

  async getAutoMLSearchStatus() {
    const response = await this.client.get('/automl/search/status');
    return response.data;
  }

  // ============================================================================
  // HyperSync Management
  // ============================================================================

  async getHyperSyncProposals(statusFilter?: string): Promise<HyperSyncProposal[]> {
    const params = statusFilter ? { status: statusFilter } : {};
    const response = await this.client.get('/hypersync/proposals', { params });
    return response.data.proposals || response.data;
  }

  async getHyperSyncProposal(proposalId: string): Promise<HyperSyncProposal> {
    const response = await this.client.get(`/hypersync/proposals/${proposalId}`);
    return response.data;
  }

  async approveHyperSyncProposal(proposalId: string, approved: boolean, reason: string) {
    const response = await this.client.post(`/hypersync/proposals/${proposalId}/approve`, {
      proposal_id: proposalId,
      approved,
      reason,
    });
    return response.data;
  }

  async getHyperSyncHistory(limit: number = 50) {
    const response = await this.client.get('/hypersync/history', { params: { limit } });
    return response.data;
  }

  // ============================================================================
  // API Key Management
  // ============================================================================

  async listAPIKeys(): Promise<APIKey[]> {
    const response = await this.client.get('/api-keys');
    return response.data.api_keys || response.data;
  }

  async createAPIKey(serviceName: string, expiresInDays?: number) {
    const response = await this.client.post('/api-keys', {
      service_name: serviceName,
      expires_in_days: expiresInDays,
    });
    return response.data;
  }

  async rotateAPIKey(keyId: string) {
    const response = await this.client.post(`/api-keys/${keyId}/rotate`);
    return response.data;
  }

  async revokeAPIKey(keyId: string, reason: string) {
    const response = await this.client.post(`/api-keys/${keyId}/revoke`, {
      key_id: keyId,
      reason,
    });
    return response.data;
  }

  // ============================================================================
  // JWT Key Management
  // ============================================================================

  async getJWTStatus() {
    const response = await this.client.get('/jwt/status');
    return response.data;
  }

  async rotateJWTKey() {
    const response = await this.client.post('/jwt/rotate');
    return response.data;
  }

  async listJWTKeys(): Promise<JWTKey[]> {
    const response = await this.client.get('/jwt/keys');
    return response.data.keys || response.data;
  }

  async invalidateJWTKey(kid: string, reason: string) {
    const response = await this.client.post(`/jwt/keys/${kid}/invalidate`, {
      reason,
    });
    return response.data;
  }

  // ============================================================================
  // System Health
  // ============================================================================

  async getSystemHealth(): Promise<SystemHealth> {
    const response = await this.client.get('/health/system');
    return response.data;
  }

  async getHealthMetrics() {
    const response = await this.client.get('/health/metrics');
    return response.data;
  }

  // ============================================================================
  // Audit Logs
  // ============================================================================

  async getAuditLogs(filters?: {
    event_type?: string;
    username?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
    offset?: number;
  }): Promise<AuditLogEvent[]> {
    const response = await this.client.get('/audit/logs', { params: filters });
    return response.data.events || response.data;
  }

  async getAuditStats(startTime?: string, endTime?: string) {
    const params = { start_time: startTime, end_time: endTime };
    const response = await this.client.get('/audit/stats', { params });
    return response.data;
  }

  async getAuditEventTypes() {
    const response = await this.client.get('/audit/event-types');
    return response.data;
  }
}

// ============================================================================
// Singleton instance (optional)
// ============================================================================

let adminClient: AdminAPIClient | null = null;

export function getAdminClient(token?: string): AdminAPIClient {
  if (!adminClient) {
    adminClient = new AdminAPIClient(token);
  } else if (token) {
    adminClient.setToken(token);
  }
  return adminClient;
}

export default AdminAPIClient;
