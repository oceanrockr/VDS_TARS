import { useState, useEffect } from 'react'
import { apiClient } from '@/lib/api'
import type { Conversation } from '@/types'
import { formatDistanceToNow } from 'date-fns'

interface SidebarProps {
  currentConversationId?: string
  onSelectConversation: (conversationId: string) => void
  onNewConversation: () => void
  onLogout: () => void
}

export function Sidebar({
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onLogout,
}: SidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showMenu, setShowMenu] = useState(false)

  useEffect(() => {
    loadConversations()
  }, [])

  const loadConversations = async () => {
    try {
      setIsLoading(true)
      const data = await apiClient.listConversations(50)
      setConversations(data)
    } catch (error) {
      console.error('Failed to load conversations:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDeleteConversation = async (conversationId: string) => {
    if (!confirm('Delete this conversation?')) return

    try {
      await apiClient.deleteConversation(conversationId)
      setConversations((prev) => prev.filter((c) => c.id !== conversationId))
    } catch (error) {
      console.error('Failed to delete conversation:', error)
    }
  }

  return (
    <div className="w-64 bg-tars-dark border-r border-gray-700 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <button
          onClick={onNewConversation}
          className="w-full px-4 py-2 bg-tars-primary hover:bg-blue-600 rounded-lg font-medium transition-colors"
        >
          + New Chat
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-4 text-center text-gray-500">Loading...</div>
        ) : conversations.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            No conversations yet
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group p-3 rounded-lg cursor-pointer transition-colors ${
                  currentConversationId === conv.id
                    ? 'bg-tars-darker border border-tars-primary'
                    : 'hover:bg-tars-darker'
                }`}
                onClick={() => onSelectConversation(conv.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">
                      {conv.title || 'Untitled Chat'}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {conv.messages.length} messages
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {formatDistanceToNow(new Date(conv.updatedAt))} ago
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDeleteConversation(conv.id)
                    }}
                    className="opacity-0 group-hover:opacity-100 ml-2 p-1 text-gray-400 hover:text-red-400 transition-opacity"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer Menu */}
      <div className="relative border-t border-gray-700 p-4">
        <button
          onClick={() => setShowMenu(!showMenu)}
          className="w-full flex items-center gap-3 p-2 hover:bg-tars-darker rounded-lg transition-colors"
        >
          <div className="w-8 h-8 rounded-full bg-tars-primary flex items-center justify-center text-sm font-bold">
            U
          </div>
          <div className="flex-1 text-left">
            <p className="text-sm font-medium">User</p>
            <p className="text-xs text-gray-400">Client</p>
          </div>
          <svg
            className="w-4 h-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"
            />
          </svg>
        </button>

        {showMenu && (
          <div className="absolute bottom-full mb-2 left-4 right-4 bg-tars-darker border border-gray-700 rounded-lg shadow-lg overflow-hidden">
            <button
              onClick={onLogout}
              className="w-full px-4 py-2 text-left text-sm hover:bg-tars-dark transition-colors text-red-400"
            >
              Logout
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
