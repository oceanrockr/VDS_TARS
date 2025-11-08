// T.A.R.S. Type Definitions

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  conversationId?: string
  sources?: SourceReference[]
  metadata?: MessageMetadata
}

export interface MessageMetadata {
  retrievalTimeMs?: number
  generationTimeMs?: number
  totalTimeMs?: number
  sourcesCount?: number
  model?: string
}

export interface SourceReference {
  documentId: string
  chunkId: string
  fileName: string
  filePath: string
  chunkIndex: number
  similarityScore: number
  excerpt: string
  pageNumber?: number
}

export interface Conversation {
  id: string
  clientId: string
  messages: Message[]
  createdAt: string
  updatedAt: string
  title?: string
}

export interface DocumentMetadata {
  documentId: string
  fileName: string
  filePath: string
  fileType: string
  fileSize: number
  indexedAt: string
  chunkCount: number
  tokenCount: number
  pageCount?: number
}

export interface RAGQueryRequest {
  query: string
  topK?: number
  relevanceThreshold?: number
  includeSources?: boolean
  rerank?: boolean
  conversationId?: string
}

export interface RAGQueryResponse {
  query: string
  answer: string
  sources: SourceReference[]
  contextUsed: string
  totalTokens: number
  retrievalTimeMs: number
  generationTimeMs: number
  totalTimeMs: number
  model: string
  relevanceScores: number[]
}

export interface SystemMetrics {
  cpuPercent: number
  memoryPercent: number
  memoryUsedMb: number
  memoryTotalMb: number
  gpuPercent?: number
  gpuMemoryPercent?: number
  gpuMemoryUsedMb?: number
  gpuMemoryTotalMb?: number
  gpuName?: string
  documentsIndexed: number
  chunksStored: number
  queriesProcessed: number
  averageRetrievalTimeMs: number
  timestamp: string
}

export interface CollectionStats {
  collectionName: string
  documentCount: number
  chunkCount: number
  totalSize: number
  lastUpdated: string
}

export interface WebSocketMessage {
  type: 'chat' | 'rag_token' | 'rag_sources' | 'rag_complete' | 'error' | 'connection'
  content?: string
  token?: string
  conversationId?: string
  useRag?: boolean
  ragTopK?: number
  ragThreshold?: number
  sources?: SourceReference[]
  retrievalTimeMs?: number
  generationTimeMs?: number
  totalTimeMs?: number
  sourcesCount?: number
  totalTokens?: number
  hasSources?: boolean
  timestamp?: string
  error?: string
  clientId?: string
}

export interface UploadProgress {
  fileName: string
  progress: number
  status: 'uploading' | 'processing' | 'complete' | 'error'
  error?: string
}

export interface ConnectionStatus {
  api: 'connected' | 'disconnected' | 'error'
  websocket: 'connected' | 'connecting' | 'disconnected' | 'error'
  lastPing?: string
}
