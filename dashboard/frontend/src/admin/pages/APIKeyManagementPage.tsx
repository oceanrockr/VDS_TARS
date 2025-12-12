/**
 * T.A.R.S. API Key Management Page
 * Admin interface for managing API keys
 * Phase 12 Part 2
 */

import React, { useState, useEffect } from 'react';
import { getAdminClient, APIKey } from '../../api/admin';
import { useAuth } from '../../hooks/useAuth';

const APIKeyManagementPage: React.FC = () => {
  const { token } = useAuth();
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [serviceName, setServiceName] = useState('');
  const [expiresInDays, setExpiresInDays] = useState<number | undefined>(undefined);
  const [newKeyData, setNewKeyData] = useState<any>(null);

  const adminClient = getAdminClient(token);

  useEffect(() => {
    fetchAPIKeys();
  }, []);

  const fetchAPIKeys = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await adminClient.listAPIKeys();
      setApiKeys(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch API keys');
      console.error('Error fetching API keys:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateAPIKey = async () => {
    if (!serviceName.trim()) {
      alert('Please provide a service name');
      return;
    }

    try {
      const response = await adminClient.createAPIKey(serviceName, expiresInDays);
      setNewKeyData(response);
      setServiceName('');
      setExpiresInDays(undefined);
      fetchAPIKeys();
    } catch (err: any) {
      alert(`Failed to create API key: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleRotateAPIKey = async (keyId: string) => {
    if (!confirm(`Are you sure you want to rotate the key for ${keyId}?`)) {
      return;
    }

    try {
      const response = await adminClient.rotateAPIKey(keyId);
      setNewKeyData(response);
      fetchAPIKeys();
    } catch (err: any) {
      alert(`Failed to rotate API key: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleRevokeAPIKey = async (keyId: string) {
    const reason = prompt(`Please provide a reason for revoking ${keyId}:`);
    if (!reason) {
      return;
    }

    try {
      await adminClient.revokeAPIKey(keyId, reason);
      alert(`API key ${keyId} revoked successfully`);
      fetchAPIKeys();
    } catch (err: any) {
      alert(`Failed to revoke API key: ${err.response?.data?.detail || err.message}`);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading API keys...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">{error}</p>
        <button
          onClick={fetchAPIKeys}
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
        <h1 className="text-3xl font-bold text-gray-900">API Key Management</h1>
        <div className="flex gap-2">
          <button
            onClick={fetchAPIKeys}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            Refresh
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Create API Key
          </button>
        </div>
      </div>

      {/* New Key Display Modal */}
      {newKeyData && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4">
            <h2 className="text-xl font-bold text-gray-900 mb-4">API Key Created</h2>
            <div className="bg-yellow-50 border border-yellow-200 rounded p-4 mb-4">
              <p className="text-sm text-yellow-800 font-medium mb-2">
                ⚠️ Save this key now! It will not be shown again.
              </p>
            </div>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Service Name
                </label>
                <p className="font-mono text-sm bg-gray-50 p-2 rounded">
                  {newKeyData.service_name}
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    readOnly
                    value={newKeyData.api_key || newKeyData.new_api_key}
                    className="flex-1 font-mono text-sm bg-gray-50 p-2 rounded border border-gray-300"
                  />
                  <button
                    onClick={() => copyToClipboard(newKeyData.api_key || newKeyData.new_api_key)}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                  >
                    Copy
                  </button>
                </div>
              </div>
            </div>
            <div className="mt-6 flex justify-end">
              <button
                onClick={() => setNewKeyData(null)}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create API Key Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Create New API Key</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Service Name
                </label>
                <input
                  type="text"
                  value={serviceName}
                  onChange={(e) => setServiceName(e.target.value)}
                  placeholder="e.g., My Service"
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Expires In (Days) - Optional
                </label>
                <input
                  type="number"
                  value={expiresInDays || ''}
                  onChange={(e) =>
                    setExpiresInDays(e.target.value ? parseInt(e.target.value) : undefined)
                  }
                  placeholder="Leave empty for no expiration"
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
            <div className="mt-6 flex gap-2 justify-end">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setServiceName('');
                  setExpiresInDays(undefined);
                }}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  handleCreateAPIKey();
                  setShowCreateModal(false);
                }}
                disabled={!serviceName.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* API Keys Table */}
      <div className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Service Name
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Key ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Created
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Last Used
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {apiKeys.map((key) => (
              <tr key={key.key_id}>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {key.service_name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                  {key.key_id}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {new Date(key.created_at).toLocaleString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {key.last_used ? new Date(key.last_used).toLocaleString() : 'Never'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span
                    className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      key.is_active
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {key.is_active ? 'Active' : 'Revoked'}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium space-x-2">
                  {key.is_active && (
                    <>
                      <button
                        onClick={() => handleRotateAPIKey(key.key_id)}
                        className="text-yellow-600 hover:text-yellow-900"
                      >
                        Rotate
                      </button>
                      <button
                        onClick={() => handleRevokeAPIKey(key.key_id)}
                        className="text-red-600 hover:text-red-900"
                      >
                        Revoke
                      </button>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {apiKeys.length === 0 && (
        <div className="text-center py-12 text-gray-500">No API keys found</div>
      )}
    </div>
  );
};

export default APIKeyManagementPage;
