#!/usr/bin/env bash
# Stop all ECG Anomaly Detection services on macOS/Linux.
for port in 3000 3003 8000; do
    pids=$(lsof -ti tcp:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            echo "Stopping PID $pid on port $port"
            kill -9 $pid 2>/dev/null || true
        done
    fi
done
echo "All services stopped."
