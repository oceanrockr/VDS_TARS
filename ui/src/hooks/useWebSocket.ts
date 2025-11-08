import { useEffect, useState, useCallback, useRef } from 'react'
import { wsClient, type WebSocketEventHandler } from '@/lib/websocket'
import type { WebSocketMessage, Message, SourceReference } from '@/types'

interface UseWebSocketReturn {
  isConnected: boolean
  sendMessage: (
    content: string,
    conversationId?: string,
    useRag?: boolean,
    ragTopK?: number,
    ragThreshold?: number
  ) => void
  messages: Message[]
  currentStreamingMessage: string
  currentSources: SourceReference[]
  isStreaming: boolean
  connect: (token: string) => Promise<void>
  disconnect: () => void
}

export function useWebSocket(): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState('')
  const [currentSources, setCurrentSources] = useState<SourceReference[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const currentMessageRef = useRef<Partial<Message>>({})

  const handleConnection: WebSocketEventHandler = useCallback((msg) => {
    setIsConnected(msg.content === 'connected')
  }, [])

  const handleRAGSources: WebSocketEventHandler = useCallback((msg) => {
    if (msg.sources) {
      setCurrentSources(msg.sources)
    }
  }, [])

  const handleRAGToken: WebSocketEventHandler = useCallback((msg) => {
    if (msg.token) {
      setIsStreaming(true)
      setCurrentStreamingMessage((prev) => prev + msg.token)
    }
  }, [])

  const handleRAGComplete: WebSocketEventHandler = useCallback((msg) => {
    setIsStreaming(false)

    // Create complete assistant message
    const assistantMessage: Message = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      role: 'assistant',
      content: currentStreamingMessage,
      timestamp: new Date().toISOString(),
      conversationId: msg.conversationId,
      sources: currentSources.length > 0 ? currentSources : undefined,
      metadata: {
        retrievalTimeMs: msg.retrievalTimeMs,
        generationTimeMs: msg.generationTimeMs,
        totalTimeMs: msg.totalTimeMs,
        sourcesCount: msg.sourcesCount,
      },
    }

    setMessages((prev) => [...prev, assistantMessage])
    setCurrentStreamingMessage('')
    setCurrentSources([])
  }, [currentStreamingMessage, currentSources])

  const handleError: WebSocketEventHandler = useCallback((msg) => {
    console.error('[WebSocket Error]:', msg.error)
    setIsStreaming(false)
    setCurrentStreamingMessage('')

    // Add error message to chat
    const errorMessage: Message = {
      id: `msg_${Date.now()}_error`,
      role: 'assistant',
      content: `Error: ${msg.error || 'Unknown error occurred'}`,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, errorMessage])
  }, [])

  useEffect(() => {
    // Register event handlers
    wsClient.on('connection', handleConnection)
    wsClient.on('rag_sources', handleRAGSources)
    wsClient.on('rag_token', handleRAGToken)
    wsClient.on('rag_complete', handleRAGComplete)
    wsClient.on('error', handleError)

    return () => {
      // Cleanup event handlers
      wsClient.off('connection', handleConnection)
      wsClient.off('rag_sources', handleRAGSources)
      wsClient.off('rag_token', handleRAGToken)
      wsClient.off('rag_complete', handleRAGComplete)
      wsClient.off('error', handleError)
    }
  }, [handleConnection, handleRAGSources, handleRAGToken, handleRAGComplete, handleError])

  const connect = useCallback(async (token: string) => {
    try {
      await wsClient.connect(token)
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      throw error
    }
  }, [])

  const disconnect = useCallback(() => {
    wsClient.disconnect()
  }, [])

  const sendMessage = useCallback(
    (
      content: string,
      conversationId?: string,
      useRag: boolean = false,
      ragTopK: number = 5,
      ragThreshold: number = 0.7
    ) => {
      // Add user message to chat
      const userMessage: Message = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        role: 'user',
        content,
        timestamp: new Date().toISOString(),
        conversationId,
      }
      setMessages((prev) => [...prev, userMessage])

      // Send via WebSocket
      wsClient.sendMessage(content, conversationId, useRag, ragTopK, ragThreshold)

      // Reset streaming state
      setCurrentStreamingMessage('')
      setCurrentSources([])
    },
    []
  )

  return {
    isConnected,
    sendMessage,
    messages,
    currentStreamingMessage,
    currentSources,
    isStreaming,
    connect,
    disconnect,
  }
}
