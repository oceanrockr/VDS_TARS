import { useState, useEffect } from 'react'
import { apiClient } from '@/lib/api'
import type { SystemMetrics, CollectionStats } from '@/types'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

export function MetricsDashboard() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [stats, setStats] = useState<CollectionStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadMetrics()
    const interval = setInterval(loadMetrics, 5000) // Refresh every 5 seconds

    return () => clearInterval(interval)
  }, [])

  const loadMetrics = async () => {
    try {
      const [metricsData, statsData] = await Promise.all([
        apiClient.getSystemMetrics(),
        apiClient.ragStats(),
      ])

      setMetrics(metricsData)
      setStats(statsData)
      setError(null)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400">Loading metrics...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="p-4 bg-red-900/30 border border-red-700 rounded-lg text-red-400">
          Failed to load metrics: {error}
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6 overflow-y-auto">
      <h2 className="text-2xl font-bold text-white">System Metrics</h2>

      {/* System Resources */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="CPU Usage"
          value={`${metrics?.cpuPercent.toFixed(1)}%`}
          color="blue"
        />
        <MetricCard
          title="Memory Usage"
          value={`${metrics?.memoryPercent.toFixed(1)}%`}
          subtitle={`${metrics?.memoryUsedMb.toFixed(0)} / ${metrics?.memoryTotalMb.toFixed(0)} MB`}
          color="purple"
        />
        {metrics?.gpuPercent !== undefined && (
          <MetricCard
            title="GPU Usage"
            value={`${metrics.gpuPercent.toFixed(1)}%`}
            subtitle={metrics.gpuName}
            color="green"
          />
        )}
        <MetricCard
          title="Avg Retrieval"
          value={`${metrics?.averageRetrievalTimeMs.toFixed(0)}ms`}
          color="orange"
        />
      </div>

      {/* Document Stats */}
      <div className="bg-tars-dark p-6 rounded-lg border border-gray-700">
        <h3 className="text-lg font-bold text-white mb-4">Document Collection</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-gray-400 text-sm">Documents Indexed</p>
            <p className="text-2xl font-bold text-white">
              {stats?.documentCount.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Total Chunks</p>
            <p className="text-2xl font-bold text-white">
              {stats?.chunkCount.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Queries Processed</p>
            <p className="text-2xl font-bold text-white">
              {metrics?.queriesProcessed.toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      {/* GPU Memory (if available) */}
      {metrics?.gpuMemoryPercent !== undefined && (
        <div className="bg-tars-dark p-6 rounded-lg border border-gray-700">
          <h3 className="text-lg font-bold text-white mb-4">GPU Memory</h3>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-300">Usage</span>
              <span className="text-white font-medium">
                {metrics.gpuMemoryUsedMb?.toFixed(0)} / {metrics.gpuMemoryTotalMb?.toFixed(0)} MB
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div
                className="h-3 bg-gradient-to-r from-tars-accent to-tars-primary rounded-full transition-all"
                style={{ width: `${metrics.gpuMemoryPercent}%` }}
              />
            </div>
            <div className="text-right text-xs text-gray-400">
              {metrics.gpuMemoryPercent.toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Collection Info */}
      {stats && (
        <div className="bg-tars-dark p-6 rounded-lg border border-gray-700">
          <h3 className="text-lg font-bold text-white mb-4">Collection Details</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Collection Name</span>
              <span className="text-white font-mono">{stats.collectionName}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Last Updated</span>
              <span className="text-white">
                {new Date(stats.lastUpdated).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Size</span>
              <span className="text-white">
                {(stats.totalSize / 1024 / 1024).toFixed(2)} MB
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface MetricCardProps {
  title: string
  value: string
  subtitle?: string
  color: 'blue' | 'purple' | 'green' | 'orange'
}

function MetricCard({ title, value, subtitle, color }: MetricCardProps) {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    purple: 'from-purple-500 to-purple-600',
    green: 'from-green-500 to-green-600',
    orange: 'from-orange-500 to-orange-600',
  }

  return (
    <div className="bg-tars-dark p-4 rounded-lg border border-gray-700">
      <p className="text-gray-400 text-sm mb-2">{title}</p>
      <p className={`text-2xl font-bold bg-gradient-to-r ${colorClasses[color]} bg-clip-text text-transparent`}>
        {value}
      </p>
      {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
    </div>
  )
}
