import { useState, useEffect, useCallback } from 'react'
import { apiClient } from '@/lib/api'

const TOKEN_STORAGE_KEY = 'tars_auth_token'
const CLIENT_ID_STORAGE_KEY = 'tars_client_id'

export function useAuth() {
  const [token, setToken] = useState<string | null>(null)
  const [clientId, setClientId] = useState<string | null>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // Load token from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem(TOKEN_STORAGE_KEY)
    const storedClientId = localStorage.getItem(CLIENT_ID_STORAGE_KEY)

    if (storedToken && storedClientId) {
      setToken(storedToken)
      setClientId(storedClientId)
      setIsAuthenticated(true)
      apiClient.setToken(storedToken)
    }

    setIsLoading(false)
  }, [])

  const login = useCallback(async (clientIdInput: string) => {
    try {
      setIsLoading(true)
      const newToken = await apiClient.authenticate(clientIdInput)

      setToken(newToken)
      setClientId(clientIdInput)
      setIsAuthenticated(true)

      // Store in localStorage
      localStorage.setItem(TOKEN_STORAGE_KEY, newToken)
      localStorage.setItem(CLIENT_ID_STORAGE_KEY, clientIdInput)

      return true
    } catch (error) {
      console.error('Authentication failed:', error)
      return false
    } finally {
      setIsLoading(false)
    }
  }, [])

  const logout = useCallback(() => {
    setToken(null)
    setClientId(null)
    setIsAuthenticated(false)

    apiClient.clearToken()
    localStorage.removeItem(TOKEN_STORAGE_KEY)
    localStorage.removeItem(CLIENT_ID_STORAGE_KEY)
  }, [])

  return {
    token,
    clientId,
    isAuthenticated,
    isLoading,
    login,
    logout,
  }
}
