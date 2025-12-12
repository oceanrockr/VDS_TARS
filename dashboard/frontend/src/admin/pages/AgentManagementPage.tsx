/**
 * T.A.R.S. Agent Management Page
 * Admin interface for managing RL agents
 * Phase 12 Part 2
 */

import React, { useState, useEffect } from 'react';
import { getAdminClient, Agent } from '../../api/admin';
import { useAuth } from '../../hooks/useAuth';

const AgentManagementPage: React.FC = () => {
  const { token } = useAuth();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [reloadReason, setReloadReason] = useState('');
  const [modelVersion, setModelVersion] = useState('');
  const [promoteReason, setPromoteReason] = useState('');

  const adminClient = getAdminClient(token);

  useEffect(() => {
    fetchAgents();
  }, []);

  const fetchAgents = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await adminClient.getAllAgents();
      setAgents(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch agents');
      console.error('Error fetching agents:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReloadAgent = async (agentId: string) => {
    if (!reloadReason.trim()) {
      alert('Please provide a reason for reloading');
      return;
    }

    try {
      await adminClient.reloadAgent(agentId, reloadReason);
      alert(`Agent ${agentId} reloaded successfully`);
      setReloadReason('');
      fetchAgents();
    } catch (err: any) {
      alert(`Failed to reload agent: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handlePromoteModel = async (agentId: string) => {
    if (!modelVersion.trim() || !promoteReason.trim()) {
      alert('Please provide model version and reason');
      return;
    }

    try {
      await adminClient.promoteModel(agentId, modelVersion, promoteReason);
      alert(`Model promoted successfully for agent ${agentId}`);
      setModelVersion('');
      setPromoteReason('');
      fetchAgents();
    } catch (err: any) {
      alert(`Failed to promote model: ${err.response?.data?.detail || err.message}`);
    }
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case 'active':
        return 'text-green-600 bg-green-100';
      case 'training':
        return 'text-blue-600 bg-blue-100';
      case 'inactive':
        return 'text-gray-600 bg-gray-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading agents...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">{error}</p>
        <button
          onClick={fetchAgents}
          className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Agent Management</h1>
        <button
          onClick={fetchAgents}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {agents.map((agent) => (
          <div
            key={agent.agent_id}
            className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden"
          >
            <div className="p-6">
              {/* Header */}
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">{agent.agent_id}</h2>
                  <p className="text-sm text-gray-500">{agent.agent_type}</p>
                </div>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${getStateColor(
                    agent.state
                  )}`}
                >
                  {agent.state}
                </span>
              </div>

              {/* Performance Metrics */}
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-xs text-gray-500 uppercase">Reward</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {agent.performance_metrics?.reward?.toFixed(2) || 'N/A'}
                  </p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-xs text-gray-500 uppercase">Loss</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {agent.performance_metrics?.loss?.toFixed(4) || 'N/A'}
                  </p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-xs text-gray-500 uppercase">Entropy</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {agent.performance_metrics?.entropy?.toFixed(3) || 'N/A'}
                  </p>
                </div>
              </div>

              {/* Model Version */}
              <div className="mb-4">
                <p className="text-sm text-gray-500">
                  Model Version:{' '}
                  <span className="font-mono text-gray-900">{agent.model_version || 'N/A'}</span>
                </p>
              </div>

              {/* Actions */}
              <div className="space-y-3">
                {/* Reload Agent */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="Reason for reload"
                    value={selectedAgent?.agent_id === agent.agent_id ? reloadReason : ''}
                    onChange={(e) => {
                      setSelectedAgent(agent);
                      setReloadReason(e.target.value);
                    }}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    onClick={() => handleReloadAgent(agent.agent_id)}
                    disabled={
                      selectedAgent?.agent_id !== agent.agent_id || !reloadReason.trim()
                    }
                    className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
                  >
                    Reload
                  </button>
                </div>

                {/* Promote Model */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="Model version"
                    value={selectedAgent?.agent_id === agent.agent_id ? modelVersion : ''}
                    onChange={(e) => {
                      setSelectedAgent(agent);
                      setModelVersion(e.target.value);
                    }}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <input
                    type="text"
                    placeholder="Reason"
                    value={selectedAgent?.agent_id === agent.agent_id ? promoteReason : ''}
                    onChange={(e) => {
                      setSelectedAgent(agent);
                      setPromoteReason(e.target.value);
                    }}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    onClick={() => handlePromoteModel(agent.agent_id)}
                    disabled={
                      selectedAgent?.agent_id !== agent.agent_id ||
                      !modelVersion.trim() ||
                      !promoteReason.trim()
                    }
                    className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
                  >
                    Promote
                  </button>
                </div>
              </div>

              {/* Hyperparameters (collapsible) */}
              <details className="mt-4">
                <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
                  View Hyperparameters
                </summary>
                <pre className="mt-2 p-3 bg-gray-50 rounded text-xs overflow-auto max-h-48">
                  {JSON.stringify(agent.hyperparameters, null, 2)}
                </pre>
              </details>
            </div>
          </div>
        ))}
      </div>

      {agents.length === 0 && (
        <div className="text-center py-12 text-gray-500">No agents found</div>
      )}
    </div>
  );
};

export default AgentManagementPage;
