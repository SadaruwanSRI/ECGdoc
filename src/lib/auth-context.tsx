'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { api, getToken, clearToken } from '@/lib/api'

type User = {
  id: string
  email: string
  name?: string
  role: string
}

type AuthContextType = {
  user: User | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, name: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  // Restore session from localStorage on mount
  useEffect(() => {
    const token = getToken()
    if (!token) {
      Promise.resolve().then(() => setLoading(false))
      return
    }
    // Token format: "<hex_json>.<hex_sig>" — decode hex (not base64).
    try {
      const [bodyHex] = token.split('.')
      // Decode hex to string, then parse JSON
      const bodyJson = decodeURIComponent(
        bodyHex.replace(/(..)/g, '%$1')
      )
      const body = JSON.parse(bodyJson)
      Promise.resolve().then(() => {
        setUser({
          id: body.sub || '',
          email: body.email || '',
          name: body.name || '',
          role: body.role || 'clinician',
        })
        setLoading(false)
      })
    } catch {
      Promise.resolve().then(() => setLoading(false))
    }
  }, [])

  const login = async (email: string, password: string) => {
    const r = await api.login(email, password)
    if (r.ok && r.data?.user) {
      setUser(r.data.user)
    } else {
      throw new Error(r.detail || 'Login failed')
    }
  }

  const register = async (email: string, name: string, password: string) => {
    const r = await api.register(email, name, password)
    if (r.ok && r.data) {
      // After register, we don't have full user info — login to get it
      await login(email, password)
    } else {
      throw new Error(r.detail || 'Registration failed')
    }
  }

  const logout = () => {
    clearToken()
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
