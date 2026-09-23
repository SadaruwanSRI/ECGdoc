/**
 * API client for the ECG Anomaly Detection FastAPI backend.
 *
 * Environment detection:
 * - Sandbox: page served from port 81 (Caddy gateway). API requests go through
 *   Caddy with XTransformPort=8000 in the query string.
 * - Local: page served from port 3000 (Next.js dev server). API requests go
 *   directly to http://localhost:8000, and WebSocket to ws://localhost:3003.
 */

const API_PORT = Number(process.env.NEXT_PUBLIC_API_PORT ?? '8000')
const WS_PORT = Number(process.env.NEXT_PUBLIC_WS_PORT ?? '3003')

/**
 * Detect whether we're running in the sandbox (Caddy on port 81) or locally
 * (Next.js on port 3000). Returns true if we're in the sandbox.
 */
function isSandbox(): boolean {
  if (typeof window === 'undefined') return false
  // In the sandbox, the page is served from port 81 (Caddy).
  // Locally, the page is served from port 3000 (Next.js dev server).
  return window.location.port === '81'
}

/**
 * Build the base URL for API requests.
 * - Sandbox: relative URL with XTransformPort query param (Caddy routes it)
 * - Local: absolute URL to http://localhost:8000
 */
function apiBase(): string {
  if (isSandbox()) {
    return ''  // relative URL — Caddy will route via XTransformPort
  }
  return `http://localhost:${API_PORT}`
}

/** Build a URL for an API request. */
function apiUrl(path: string): string {
  if (isSandbox()) {
    const sep = path.includes('?') ? '&' : '?'
    return `${path}${sep}XTransformPort=${API_PORT}`
  }
  return `${apiBase()}${path}`
}

/** Build a WebSocket URL for socket.io connections. */
export function wsUrl(): string {
  if (isSandbox()) {
    return `/?XTransformPort=${WS_PORT}`
  }
  return `http://localhost:${WS_PORT}`
}

// Token is stored in localStorage after login
export function getToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem('ecg_token')
}

export function setToken(token: string) {
  if (typeof window === 'undefined') return
  localStorage.setItem('ecg_token', token)
}

export function clearToken() {
  if (typeof window === 'undefined') return
  localStorage.removeItem('ecg_token')
}

async function request<T = any>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getToken()
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  }
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (options.body && typeof options.body === 'string') {
    headers['Content-Type'] = 'application/json'
  }

  const resp = await fetch(apiUrl(path), { ...options, headers })
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const err = await resp.json()
      detail = err.detail || detail
    } catch {}

    // On 401 (Unauthorized), clear the token and reload to login screen
    if (resp.status === 401) {
      clearToken()
      if (typeof window !== 'undefined') {
        // Reload the page so the auth context picks up the cleared token
        window.location.reload()
      }
    }

    throw new Error(detail)
  }
  return resp.json()
}

// ---------- Auth ----------
export const api = {
  async register(email: string, name: string, password: string) {
    const r = await request<any>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, name, password }),
    })
    if (r.ok && r.data?.token) setToken(r.data.token)
    return r
  },

  async login(email: string, password: string) {
    const r = await request<any>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })
    if (r.ok && r.data?.token) setToken(r.data.token)
    return r
  },

  logout() {
    clearToken()
  },

  // ---------- Models ----------
  async listModels() {
    return request<any>('/api/models')
  },

  async getModel(id: string) {
    return request<any>(`/api/models/${id}`)
  },

  async compareModels(ids: string[]) {
    return request<any>(`/api/models/compare?ids=${ids.join(',')}`)
  },

  async archiveModel(id: string) {
    return request<any>(`/api/models/${id}`, { method: 'DELETE' })
  },

  async deleteModel(id: string) {
    return request<any>(`/api/models/${id}`, { method: 'DELETE' })
  },

  async deleteSession(id: string) {
    return request<any>(`/api/sessions/${id}`, { method: 'DELETE' })
  },

  // ---------- Training ----------
  async startTraining(config: any) {
    return request<any>('/api/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  },

  async startClassifierTraining(config: any) {
    return request<any>('/api/training/start-classifier', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  },

  async getTrainingRun(runId: string) {
    return request<any>(`/api/training/${runId}`)
  },

  async stopTrainingRun(runId: string) {
    return request<any>(`/api/training/${runId}/stop`, { method: 'POST' })
  },

  async listTrainingRuns() {
    return request<any>('/api/training')
  },

  async getTrainingArchitectures() {
    return request<any>('/api/training/architectures')
  },

  async evaluateMitbih(config: any) {
    return request<any>('/api/evaluation/mitbih', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  },

  async listEvaluationDatasets() {
    return request<any>('/api/evaluation/datasets')
  },

  async evaluateDataset(config: any) {
    return request<any>('/api/evaluation/run', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  },

  async startEvaluation(config: any) {
    return request<any>('/api/evaluation/start', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  },

  async getEvaluationProgress(jobId: string) {
    return request<any>(`/api/evaluation/progress/${jobId}`)
  },

  async getFinalStudy() {
    return request<any>('/api/evaluation/final-study')
  },

  // ---------- Sessions ----------
  async startSession(config: any) {
    return request<any>('/api/sessions/start', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  },

  async getHealthInfo(arrhythmiaType: string) {
    return request<any>(`/api/datasets/health-info/${arrhythmiaType}`)
  },

  async listHealthInfo() {
    return request<any>('/api/datasets/health-info')
  },

  async stopSession(id: string) {
    return request<any>(`/api/sessions/${id}/stop`, { method: 'POST' })
  },

  async listSessions() {
    return request<any>('/api/sessions')
  },

  async getSession(id: string) {
    return request<any>(`/api/sessions/${id}`)
  },

  async getSessionDatapoints(id: string, limit = 2000) {
    return request<any>(`/api/sessions/${id}/datapoints?limit=${limit}`)
  },

  async listArduinoPorts() {
    return request<any>('/api/arduino/ports')
  },

  // ---------- Datasets ----------
  async listDatasets() {
    return request<any>('/api/datasets')
  },

  async listUploads() {
    return request<any>('/api/datasets/uploads')
  },

  async uploadDataset(datasetId: string, files: File[]) {
    const formData = new FormData()
    formData.append('dataset_id', datasetId)
    files.forEach(f => formData.append('files', f))
    // Note: don't set Content-Type header — browser sets it with boundary
    const token = getToken()
    const headers: Record<string, string> = {}
    if (token) headers['Authorization'] = `Bearer ${token}`
    const resp = await fetch(apiUrl('/api/datasets/upload'), {
      method: 'POST',
      headers,
      body: formData,
    })
    if (!resp.ok) {
      let detail = `${resp.status} ${resp.statusText}`
      try { const err = await resp.json(); detail = err.detail || detail } catch {}
      throw new Error(detail)
    }
    return resp.json()
  },

  async deleteUpload(datasetId: string) {
    return request<any>(`/api/datasets/uploads/${datasetId}`, { method: 'DELETE' })
  },

  // ---------- Reports ----------
  async generateReport(sessionId: string, physicianName?: string, notes?: string) {
    return request<any>('/api/reports/generate', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        physician_name: physicianName,
        notes,
      }),
    })
  },

  async downloadReport(filename: string) {
    const token = getToken()
    const headers: Record<string, string> = {}
    if (token) headers['Authorization'] = `Bearer ${token}`
    const resp = await fetch(apiUrl(`/api/reports/download/${encodeURIComponent(filename)}`), {
      headers,
    })
    if (!resp.ok) {
      let detail = `${resp.status} ${resp.statusText}`
      try { const err = await resp.json(); detail = err.detail || detail } catch {}
      throw new Error(detail)
    }
    const blobUrl = URL.createObjectURL(await resp.blob())
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = filename
    link.click()
    URL.revokeObjectURL(blobUrl)
  },
}

// ---------- WebSocket helpers ----------
import { io, Socket } from 'socket.io-client'

export function connectTraining(runId: string): Socket {
  return io(wsUrl(), {
    transports: ['websocket', 'polling'],
    forceNew: true,
    reconnection: true,
    reconnectionAttempts: 5,
    timeout: 10000,
  })
}

export function connectSession(sessionId: string): Socket {
  return io(wsUrl(), {
    transports: ['websocket', 'polling'],
    forceNew: true,
    reconnection: true,
    reconnectionAttempts: 5,
    timeout: 10000,
  })
}
