#!/bin/bash
# Double-fork daemon - completely detaches from controlling terminal
cd /home/z/my-project/backend
exec /home/z/.venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> /home/z/my-project/backend/logs/fastapi.log 2>&1
