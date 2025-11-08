import { useState } from 'react'
import { apiClient } from '@/lib/api'
import type { UploadProgress, DocumentMetadata } from '@/types'

interface DocumentUploadProps {
  onUploadComplete?: (metadata: DocumentMetadata) => void
}

export function DocumentUpload({ onUploadComplete }: DocumentUploadProps) {
  const [filePath, setFilePath] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [progress, setProgress] = useState<UploadProgress | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lastUpload, setLastUpload] = useState<DocumentMetadata | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!filePath.trim()) return

    setIsUploading(true)
    setError(null)
    setProgress({
      fileName: filePath,
      progress: 0,
      status: 'uploading',
    })

    try {
      const result = await apiClient.indexDocument(filePath.trim(), false)

      setProgress({
        fileName: filePath,
        progress: 100,
        status: 'complete',
      })

      setLastUpload(result.metadata)
      onUploadComplete?.(result.metadata)

      setTimeout(() => {
        setFilePath('')
        setProgress(null)
      }, 3000)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to index document')
      setProgress({
        fileName: filePath,
        progress: 0,
        status: 'error',
        error: err.message,
      })
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="bg-tars-dark p-6 rounded-lg border border-gray-700">
      <h3 className="text-lg font-bold text-white mb-4">Index Document</h3>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Document Path
          </label>
          <input
            type="text"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            placeholder="/path/to/document.pdf"
            disabled={isUploading}
            className="w-full px-4 py-2 bg-tars-darker border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-tars-primary disabled:opacity-50"
          />
          <p className="mt-2 text-xs text-gray-500">
            Supported: PDF, DOCX, TXT, MD, CSV
          </p>
        </div>

        {error && (
          <div className="p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        {progress && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-300">{progress.fileName}</span>
              <span
                className={`font-medium ${
                  progress.status === 'complete'
                    ? 'text-tars-accent'
                    : progress.status === 'error'
                    ? 'text-red-400'
                    : 'text-tars-primary'
                }`}
              >
                {progress.status === 'complete'
                  ? 'Complete'
                  : progress.status === 'error'
                  ? 'Error'
                  : `${progress.progress}%`}
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  progress.status === 'complete'
                    ? 'bg-tars-accent'
                    : progress.status === 'error'
                    ? 'bg-red-500'
                    : 'bg-tars-primary'
                }`}
                style={{ width: `${progress.progress}%` }}
              />
            </div>
          </div>
        )}

        {lastUpload && !progress && (
          <div className="p-3 bg-tars-accent/20 border border-tars-accent rounded-lg">
            <p className="text-sm text-tars-accent font-medium mb-2">
              Last Upload Successful
            </p>
            <div className="text-xs text-gray-300 space-y-1">
              <p>File: {lastUpload.fileName}</p>
              <p>Chunks: {lastUpload.chunkCount}</p>
              <p>Type: {lastUpload.fileType.toUpperCase()}</p>
              {lastUpload.pageCount && <p>Pages: {lastUpload.pageCount}</p>}
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={isUploading || !filePath.trim()}
          className="w-full px-4 py-2 bg-tars-primary hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
        >
          {isUploading ? 'Indexing...' : 'Index Document'}
        </button>
      </form>
    </div>
  )
}
