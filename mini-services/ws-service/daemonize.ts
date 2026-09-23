// Bun-based daemon launcher
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const serviceDir = dirname(fileURLToPath(import.meta.url))
const logFile = Bun.file(join(serviceDir, 'ws.log'))
const proc = Bun.spawn(['bun', 'run', 'index.ts'], {
  cwd: serviceDir,
  stdout: logFile,
  stderr: logFile,
  env: { ...process.env },
  detached: true,
})
proc.unref()
console.log(`Started ws-service PID: ${proc.pid}`)
