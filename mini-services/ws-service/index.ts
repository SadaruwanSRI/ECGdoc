/**
 * WebSocket service for real-time ECG streaming + training progress.
 *
 * Two namespaces of events:
 * 1. training:{run_id} — emits "progress" events pulled from FastAPI's
 *    in-memory run registry via HTTP polling (every 1s).
 * 2. session:{session_id} — emits "chunk" events pulled from FastAPI's
 *    next_live_chunk() via HTTP (every 1s, matching the chunk duration).
 *
 * Both are pull-based because FastAPI holds the single source of truth
 * (training threads + live sessions). Socket.io is the transport layer.
 *
 * Caddy route: io("/?XTransformPort=3003") from the browser.
 */
import { createServer } from 'http'
import { Server } from 'socket.io'

const FASTAPI = 'http://localhost:8000'

const httpServer = createServer()
const io = new Server(httpServer, {
  path: '/',
  cors: { origin: '*', methods: ['GET', 'POST'] },
  pingTimeout: 60000,
  pingInterval: 25000,
})

// ---------- Helper: HTTP fetch JSON from FastAPI ----------
async function fetchJSON(path: string, token?: string, method: string = 'GET', body?: any) {
  const headers: Record<string, string> = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (body) headers['Content-Type'] = 'application/json'
  const resp = await fetch(`${FASTAPI}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`HTTP ${resp.status}: ${text}`)
  }
  return resp.json()
}

// ---------- Training progress poller ----------
// run_id -> { token, lastIndex, intervalId, socketIds: Set }
const trainingSubs = new Map<string, any>()

async function pollTrainingProgress(run_id: string) {
  const sub = trainingSubs.get(run_id)
  if (!sub) return
  try {
    const data = await fetchJSON(`/api/training/${run_id}`, sub.token)
    const d = data.data || {}
    const progress = d.progress || []
    // Emit only new events
    for (let i = sub.lastIndex; i < progress.length; i++) {
      io.to(`training:${run_id}`).emit('progress', progress[i])
    }
    sub.lastIndex = progress.length

    // If terminal state, emit a final event + cleanup
    if (d.status === 'completed' || d.status === 'failed' || d.status === 'stopped') {
      io.to(`training:${run_id}`).emit('done', {
        status: d.status,
        final_result: d.final_result,
        error: d.error,
      })
      clearInterval(sub.intervalId)
      trainingSubs.delete(run_id)
    }
  } catch (e: any) {
    console.error(`[training ${run_id}] poll error:`, e.message)
  }
}

// ---------- Live session chunk poller ----------
// session_id -> { token, intervalId, socketIds: Set }
const sessionSubs = new Map<string, any>()

async function pollSessionChunk(session_id: string) {
  const sub = sessionSubs.get(session_id)
  if (!sub) return
  try {
    // Pull next chunk by calling an internal endpoint on FastAPI.
    // We piggyback on the existing GET /api/sessions/{id} for status,
    // and POST an "internal" endpoint to pull next chunk.
    //
    // Actually the cleanest: call the FastAPI /api/sessions/_internal/{id}/next
    // (defined in routes_sessions.py as next_live_chunk helper exposed via POST).
    const data = await fetchJSON(
      `/api/sessions/${session_id}/next-chunk`,
      sub.token,
      'POST',
      {},
    )
    const d = data.data || {}
    if (d.type === 'chunk') {
      io.to(`session:${session_id}`).emit('chunk', d)
    } else if (d.type === 'end') {
      io.to(`session:${session_id}`).emit('end', d)
      clearInterval(sub.intervalId)
      sessionSubs.delete(session_id)
    } else if (d.type === 'error') {
      io.to(`session:${session_id}`).emit('error', d)
      clearInterval(sub.intervalId)
      sessionSubs.delete(session_id)
    }
  } catch (e: any) {
    console.error(`[session ${session_id}] poll error:`, e.message)
  }
}

// ---------- Socket.io connection handler ----------
io.on('connection', (socket) => {
  console.log(`[ws] Client connected: ${socket.id}`)

  // ---- Training subscriptions ----
  socket.on('subscribe:training', ({ run_id, token }: { run_id: string; token: string }) => {
    if (!run_id || !token) {
      socket.emit('error', { message: 'run_id and token required' })
      return
    }
    socket.join(`training:${run_id}`)
    let sub = trainingSubs.get(run_id)
    if (!sub) {
      sub = { token, lastIndex: 0, intervalId: null, socketIds: new Set() }
      sub.intervalId = setInterval(() => pollTrainingProgress(run_id), 1000)
      trainingSubs.set(run_id, sub)
    }
    sub.socketIds.add(socket.id)
    console.log(`[ws] ${socket.id} subscribed to training ${run_id}`)
  })

  socket.on('unsubscribe:training', ({ run_id }: { run_id: string }) => {
    socket.leave(`training:${run_id}`)
    const sub = trainingSubs.get(run_id)
    if (sub) {
      sub.socketIds.delete(socket.id)
      if (sub.socketIds.size === 0) {
        clearInterval(sub.intervalId)
        trainingSubs.delete(run_id)
      }
    }
  })

  // ---- Live session subscriptions ----
  socket.on('subscribe:session', ({ session_id, token }: { session_id: string; token: string }) => {
    if (!session_id || !token) {
      socket.emit('error', { message: 'session_id and token required' })
      return
    }
    socket.join(`session:${session_id}`)
    let sub = sessionSubs.get(session_id)
    if (!sub) {
      sub = { token, intervalId: null, socketIds: new Set() }
      sub.intervalId = setInterval(() => pollSessionChunk(session_id), 1000)
      sessionSubs.set(session_id, sub)
    }
    sub.socketIds.add(socket.id)
    console.log(`[ws] ${socket.id} subscribed to session ${session_id}`)
  })

  socket.on('unsubscribe:session', ({ session_id }: { session_id: string }) => {
    socket.leave(`session:${session_id}`)
    const sub = sessionSubs.get(session_id)
    if (sub) {
      sub.socketIds.delete(socket.id)
      if (sub.socketIds.size === 0) {
        clearInterval(sub.intervalId)
        sessionSubs.delete(session_id)
      }
    }
  })

  socket.on('disconnect', () => {
    console.log(`[ws] Client disconnected: ${socket.id}`)
    // Cleanup any subs
    for (const [run_id, sub] of trainingSubs.entries()) {
      if (sub.socketIds.has(socket.id)) {
        sub.socketIds.delete(socket.id)
        if (sub.socketIds.size === 0) {
          clearInterval(sub.intervalId)
          trainingSubs.delete(run_id)
        }
      }
    }
    for (const [session_id, sub] of sessionSubs.entries()) {
      if (sub.socketIds.has(socket.id)) {
        sub.socketIds.delete(socket.id)
        if (sub.socketIds.size === 0) {
          clearInterval(sub.intervalId)
          sessionSubs.delete(session_id)
        }
      }
    }
  })
})

const PORT = 3003
httpServer.listen(PORT, () => {
  console.log(`[ws] WebSocket server running on port ${PORT}`)
})

process.on('SIGTERM', () => {
  console.log('[ws] SIGTERM received, shutting down')
  httpServer.close(() => process.exit(0))
})
