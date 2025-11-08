import { useState, useRef, useEffect } from 'react'
import type { Message, SourceReference } from '@/types'
import ReactMarkdown from 'react-markdown'
import { formatDistanceToNow } from 'date-fns'

interface ChatPanelProps {
  messages: Message[]
  currentStreamingMessage: string
  currentSources: SourceReference[]
  isStreaming: boolean
  isConnected: boolean
  onSendMessage: (content: string, useRag: boolean) => void
}

export function ChatPanel({
  messages,
  currentStreamingMessage,
  currentSources,
  isStreaming,
  isConnected,
  onSendMessage,
}: ChatPanelProps) {
  const [inputValue, setInputValue] = useState('')
  const [useRag, setUseRag] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, currentStreamingMessage])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || !isConnected) return

    onSendMessage(inputValue.trim(), useRag)
    setInputValue('')
  }

  return (
    <div className="flex flex-col h-full bg-tars-darker">
      {/* Header */}
      <div className="border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-tars-primary">T.A.R.S. Chat</h1>
            <p className="text-sm text-gray-400">
              Temporal Augmented Retrieval System
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-tars-accent' : 'bg-red-500'
              }`}
            />
            <span className="text-sm text-gray-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {/* Streaming message */}
        {isStreaming && (
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-tars-secondary flex items-center justify-center text-sm font-bold">
              AI
            </div>
            <div className="flex-1">
              {currentSources.length > 0 && (
                <div className="mb-2 p-2 bg-tars-dark rounded-lg">
                  <p className="text-xs text-gray-400 mb-1">
                    Sources ({currentSources.length})
                  </p>
                  {currentSources.map((source, idx) => (
                    <div key={idx} className="text-xs text-gray-300">
                      • {source.fileName} (score: {source.similarityScore.toFixed(2)})
                    </div>
                  ))}
                </div>
              )}
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown>{currentStreamingMessage}</ReactMarkdown>
              </div>
              <div className="flex items-center gap-1 mt-1">
                <div className="w-2 h-2 bg-tars-accent rounded-full animate-pulse" />
                <span className="text-xs text-gray-500">Streaming...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-700 p-4">
        <form onSubmit={handleSubmit} className="space-y-2">
          <div className="flex items-center gap-2 mb-2">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={useRag}
                onChange={(e) => setUseRag(e.target.checked)}
                className="rounded border-gray-600 bg-tars-dark text-tars-primary focus:ring-tars-primary"
              />
              <span className="text-gray-300">Enable RAG</span>
            </label>
            {useRag && (
              <span className="text-xs text-tars-accent">
                Document sources will be retrieved
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={
                isConnected
                  ? 'Ask a question...'
                  : 'Connecting...'
              }
              disabled={!isConnected || isStreaming}
              className="flex-1 px-4 py-2 bg-tars-dark border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-tars-primary disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!isConnected || !inputValue.trim() || isStreaming}
              className="px-6 py-2 bg-tars-primary hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
          isUser ? 'bg-tars-primary' : 'bg-tars-secondary'
        }`}
      >
        {isUser ? 'U' : 'AI'}
      </div>
      <div className="flex-1 max-w-[80%]">
        <div
          className={`p-3 rounded-lg ${
            isUser ? 'bg-tars-primary' : 'bg-tars-dark'
          }`}
        >
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 p-2 bg-tars-dark rounded-lg text-xs">
            <p className="text-gray-400 font-medium mb-1">
              Sources ({message.sources.length})
            </p>
            {message.sources.map((source, idx) => (
              <details key={idx} className="mb-1">
                <summary className="cursor-pointer text-gray-300 hover:text-white">
                  {source.fileName} (score: {source.similarityScore.toFixed(2)})
                </summary>
                <div className="mt-1 pl-4 text-gray-400 border-l-2 border-tars-primary">
                  <p className="italic">&quot;{source.excerpt}&quot;</p>
                  {source.pageNumber && (
                    <p className="mt-1">Page {source.pageNumber}</p>
                  )}
                </div>
              </details>
            ))}
          </div>
        )}

        {/* Metadata */}
        <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
          <span>{formatDistanceToNow(new Date(message.timestamp))} ago</span>
          {message.metadata?.totalTimeMs && (
            <span>• {message.metadata.totalTimeMs.toFixed(0)}ms</span>
          )}
        </div>
      </div>
    </div>
  )
}
