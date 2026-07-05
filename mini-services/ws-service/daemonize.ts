// Bun-based daemon launcher
const proc = Bun.spawn(['bun', 'run', 'index.ts'], {
  cwd: '/home/z/my-project/mini-services/ws-service',
  stdout: '/home/z/my-project/mini-services/ws-service/ws.log',
  stderr: '/home/z/my-project/mini-services/ws-service/ws.log',
  env: { ...process.env },
  detached: true,
})
proc.unref()
console.log(`Started ws-service PID: ${proc.pid}`)
