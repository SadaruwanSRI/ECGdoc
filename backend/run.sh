#!/bin/bash
cd /home/z/my-project/backend
exec /home/z/.venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> logs/fastapi.log 2>&1
