"""Daemonize a process using proper double-fork + setsid."""
import os
import sys

def daemonize_and_run():
    os.chdir('/home/z/my-project/backend')
    sys.path.insert(0, '/home/z/my-project/backend')

    # First fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)
    
    os.setsid()
    os.umask(0)
    
    # Second fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)
    
    # Redirect std streams
    sys.stdout.flush()
    sys.stderr.flush()
    log = open('/home/z/my-project/backend/logs/fastapi.log', 'a')
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    devnull = open('/dev/null', 'r')
    os.dup2(devnull.fileno(), 0)
    
    # Exec uvicorn
    os.execvp('/home/z/.venv/bin/python3', [
        '/home/z/.venv/bin/python3', '-m', 'uvicorn',
        'app.main:app', '--host', '0.0.0.0', '--port', '8000',
    ])

if __name__ == '__main__':
    daemonize_and_run()
