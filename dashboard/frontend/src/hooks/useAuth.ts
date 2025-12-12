/**
 * T.A.R.S. Authentication Hook
 * React hook for managing authentication state
 * Phase 12 Part 2
 */

import { useState, useEffect, createContext, useContext } from 'react';

export interface AuthState {
  token: string | null;
  user: {
    user_id: string;
    username: string;
    roles: string[];
    email?: string;
  } | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

const AuthContext = createContext<AuthState & {
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  setToken: (token: string) => void;
}>({
  token: null,
  user: null,
  isAuthenticated: false,
  isLoading: true,
  login: async () => {},
  logout: () => {},
  setToken: () => {},
});

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    // Return default values if provider is not available
    return {
      token: localStorage.getItem('auth_token'),
      user: null,
      isAuthenticated: !!localStorage.getItem('auth_token'),
      isLoading: false,
      login: async () => {},
      logout: () => {},
      setToken: (token: string) => localStorage.setItem('auth_token', token),
    };
  }
  return context;
};

export default useAuth;
