#!/bin/bash
# Mini-service launcher for the FastAPI backend.
# Runs the Python uvicorn process; output is captured by dev.sh.
cd /home/z/my-project/backend
exec /home/z/.venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
