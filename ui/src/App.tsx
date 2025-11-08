import { useState, useEffect } from 'react'
import { useAuth } from '@/hooks/useAuth'
import { useWebSocket } from '@/hooks/useWebSocket'
import { ChatPanel } from '@/components/ChatPanel'
import { Sidebar } from '@/components/Sidebar'
import { DocumentUpload } from '@/components/DocumentUpload'
import { MetricsDashboard } from '@/components/MetricsDashboard'

function App() {
  const { token, clientId, isAuthenticated, isLoading: authLoading, login, logout } = useAuth()
  const {
    isConnected,
    sendMessage,
    messages,
    currentStreamingMessage,
    currentSources,
    isStreaming,
    connect,
    disconnect,
  } = useWebSocket()

  const [currentConversationId, setCurrentConversationId] = useState<string>()
  const [activeTab, setActiveTab] = useState<'chat' | 'upload' | 'metrics'>('chat')
  const [loginClientId, setLoginClientId] = useState('')

  // Connect WebSocket when authenticated
  useEffect(() => {
    if (isAuthenticated && token) {
      connect(token).catch((error) => {
        console.error('Failed to connect WebSocket:', error)
      })
    } else {
      disconnect()
    }

    return () => {
      disconnect()
    }
  }, [isAuthenticated, token])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!loginClientId.trim()) return

    const success = await login(loginClientId.trim())
    if (success) {
      setLoginClientId('')
    }
  }

  const handleLogout = () => {
    logout()
    setCurrentConversationId(undefined)
  }

  const handleSendMessage = (content: string, useRag: boolean) => {
    sendMessage(
      content,
      currentConversationId || `conv_${Date.now()}`,
      useRag,
      5,
      0.7
    )
  }

  const handleNewConversation = () => {
    setCurrentConversationId(`conv_${Date.now()}`)
  }

  if (authLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-tars-darker">
        <div className="text-gray-400">Loading...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center h-screen bg-tars-darker">
        <div className="w-full max-w-md p-8 bg-tars-dark rounded-lg border border-gray-700">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-tars-primary mb-2">T.A.R.S.</h1>
            <p className="text-gray-400">Temporal Augmented Retrieval System</p>
          </div>

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Client ID
              </label>
              <input
                type="text"
                value={loginClientId}
                onChange={(e) => setLoginClientId(e.target.value)}
                placeholder="Enter client ID"
                className="w-full px-4 py-2 bg-tars-darker border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-tars-primary"
              />
            </div>

            <button
              type="submit"
              disabled={!loginClientId.trim()}
              className="w-full px-4 py-3 bg-tars-primary hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
            >
              Connect
            </button>
          </form>

          <p className="mt-6 text-xs text-gray-500 text-center">
            Version 0.2.0-alpha • Phase 4
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-tars-darker">
      {/* Sidebar */}
      <Sidebar
        currentConversationId={currentConversationId}
        onSelectConversation={setCurrentConversationId}
        onNewConversation={handleNewConversation}
        onLogout={handleLogout}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Tab Navigation */}
        <div className="border-b border-gray-700 bg-tars-dark">
          <div className="flex gap-1 p-2">
            <TabButton
              active={activeTab === 'chat'}
              onClick={() => setActiveTab('chat')}
            >
              Chat
            </TabButton>
            <TabButton
              active={activeTab === 'upload'}
              onClick={() => setActiveTab('upload')}
            >
              Upload
            </TabButton>
            <TabButton
              active={activeTab === 'metrics'}
              onClick={() => setActiveTab('metrics')}
            >
              Metrics
            </TabButton>
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === 'chat' && (
            <ChatPanel
              messages={messages}
              currentStreamingMessage={currentStreamingMessage}
              currentSources={currentSources}
              isStreaming={isStreaming}
              isConnected={isConnected}
              onSendMessage={handleSendMessage}
            />
          )}

          {activeTab === 'upload' && (
            <div className="p-6">
              <DocumentUpload />
            </div>
          )}

          {activeTab === 'metrics' && <MetricsDashboard />}
        </div>
      </div>
    </div>
  )
}

interface TabButtonProps {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}

function TabButton({ active, onClick, children }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-lg font-medium transition-colors ${
        active
          ? 'bg-tars-primary text-white'
          : 'text-gray-400 hover:text-white hover:bg-tars-darker'
      }`}
    >
      {children}
    </button>
  )
}

export default App
