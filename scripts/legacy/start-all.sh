#!/usr/bin/env bash
# ==========================================================================
# ECG Anomaly Detection — Full Stack Launcher for macOS/Linux
# ==========================================================================
# Equivalent to start-all.ps1 but for bash.
#
# Usage:
#   ./start-all.sh                # First-time setup with SQLite
#   ./start-all.sh --postgres     # Use PostgreSQL
#   ./start-all.sh --skip-setup   # Skip install steps
#   ./start-all.sh --train-demo   # Train a demo model after starting
# ==========================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
WS_SERVICE_DIR="$PROJECT_ROOT/mini-services/ws-service"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"
VENV_DIR="$BACKEND_DIR/venv"

BACKEND_PORT=8000
WS_PORT=3003
FRONTEND_PORT=3000

USE_POSTGRES=0
SKIP_SETUP=0
TRAIN_DEMO=0

for arg in "$@"; do
    case "$arg" in
        --postgres)    USE_POSTGRES=1 ;;
        --skip-setup)  SKIP_SETUP=1 ;;
        --train-demo)  TRAIN_DEMO=1 ;;
        --help|-h)
            echo "Usage: $0 [--postgres] [--skip-setup] [--train-demo]"
            exit 0 ;;
    esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
GRAY='\033[0;90m'
NC='\033[0m'

echo_step()  { echo -e "${YELLOW}==> $1${NC}"; }
echo_ok()    { echo -e "    ${GREEN}OK  $1${NC}"; }
echo_err()   { echo -e "    ${RED}ERR $1${NC}"; }

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  ECG Anomaly Detection — Full Stack Launcher${NC}"
echo -e "${CYAN}================================================================${NC}"

# ---------- Step 1: Prerequisites ----------
if [ $SKIP_SETUP -eq 0 ]; then
    echo -e "${CYAN}Step 1/5 — Verifying prerequisites${NC}"

    if ! command -v python3 &>/dev/null; then
        echo_err "python3 not found. Install Python 3.10+."
        exit 1
    fi
    echo_ok "Python3: $(python3 --version)"

    USE_BUN=0
    if command -v bun &>/dev/null; then
        echo_ok "Bun: $(bun --version)"
        USE_BUN=1
    elif command -v node &>/dev/null; then
        echo_ok "Node.js: $(node --version) (Bun not found — using npm)"
    else
        echo_err "Neither bun nor node found. Install one."
        exit 1
    fi

    if [ $USE_POSTGRES -eq 1 ]; then
        if ! command -v psql &>/dev/null; then
            echo_err "PostgreSQL not found. Install it then run: psql -U postgres -f setup-postgres.sql"
            exit 1
        fi
        echo_ok "PostgreSQL: $(psql --version)"
    else
        echo_ok "Using SQLite"
    fi
fi

# ---------- Step 2: .env ----------
if [ $SKIP_SETUP -eq 0 ]; then
    echo -e "${CYAN}Step 2/5 — Setting up .env${NC}"

    if [ ! -f "$ENV_FILE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo_ok "Created .env from .env.example"

        if [ $USE_POSTGRES -eq 1 ]; then
            sed -i.bak 's|DATABASE_URL="file:./dev.db"|DATABASE_URL="postgresql://ecg_user:ecg_password@localhost:5432/ecg_anomaly"|' "$ENV_FILE"
            rm -f "$ENV_FILE.bak"
            echo_ok "Switched DATABASE_URL to PostgreSQL"

            sed -i.bak 's|provider = "sqlite"|provider = "postgresql"|' "$PROJECT_ROOT/prisma/schema.prisma"
            rm -f "$PROJECT_ROOT/prisma/schema.prisma.bak"
            echo_ok "Updated prisma/schema.prisma to PostgreSQL"
        fi
    else
        echo_ok ".env already exists"
    fi
fi

# ---------- Step 3: Python backend ----------
if [ $SKIP_SETUP -eq 0 ]; then
    echo -e "${CYAN}Step 3/5 — Setting up Python backend${NC}"

    if [ ! -d "$VENV_DIR" ]; then
        echo_step "Creating Python venv..."
        python3 -m venv "$VENV_DIR"
        echo_ok "venv created"
    else
        echo_ok "venv already exists"
    fi

    "$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
    echo_step "Installing Python dependencies (2-5 min on first run)..."
    "$VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt" --quiet
    echo_ok "Python dependencies installed"
fi

# ---------- Step 4: Frontend + WS service ----------
if [ $SKIP_SETUP -eq 0 ]; then
    echo -e "${CYAN}Step 4/5 — Setting up frontend${NC}"

    echo_step "Installing frontend dependencies..."
    cd "$PROJECT_ROOT"
    if [ $USE_BUN -eq 1 ]; then bun install; else npm install; fi
    echo_ok "Frontend deps installed"

    echo_step "Installing WebSocket service deps..."
    cd "$WS_SERVICE_DIR"
    if [ $USE_BUN -eq 1 ]; then bun install; else
        if [ ! -d node_modules ]; then
            npm init -y
            npm install socket.io
        fi
    fi
    echo_ok "WebSocket service deps installed"
fi

# ---------- Step 5: Database ----------
if [ $SKIP_SETUP -eq 0 ]; then
    echo -e "${CYAN}Step 5/5 — Initializing database${NC}"
    cd "$PROJECT_ROOT"
    if [ $USE_BUN -eq 1 ]; then bun run db:push; else npx prisma db push; fi
    echo_ok "Database schema applied"
fi

# ---------- Launch services ----------
echo -e "${CYAN}Starting services (in background)${NC}"

# FastAPI backend
echo_step "Starting FastAPI backend (port $BACKEND_PORT)..."
cd "$BACKEND_DIR"
nohup "$VENV_DIR/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload \
    > "$BACKEND_DIR/logs/fastapi.log" 2>&1 &
BACKEND_PID=$!
echo_ok "FastAPI started (PID $BACKEND_PID)"

# WebSocket service
echo_step "Starting WebSocket service (port $WS_PORT)..."
cd "$WS_SERVICE_DIR"
if [ $USE_BUN -eq 1 ]; then
    nohup bun run index.ts > "$WS_SERVICE_DIR/ws.log" 2>&1 &
else
    nohup npx tsx index.ts > "$WS_SERVICE_DIR/ws.log" 2>&1 &
fi
WS_PID=$!
echo_ok "WebSocket service started (PID $WS_PID)"

# Wait for backend
echo_step "Waiting for backend..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:$BACKEND_PORT/health" | grep -q '"ok":true'; then
        echo_ok "Backend ready"
        break
    fi
    sleep 1
done

# Next.js frontend
echo_step "Starting Next.js frontend (port $FRONTEND_PORT)..."
cd "$PROJECT_ROOT"
if [ $USE_BUN -eq 1 ]; then
    nohup bun run dev > "$PROJECT_ROOT/frontend.log" 2>&1 &
else
    nohup npm run dev > "$PROJECT_ROOT/frontend.log" 2>&1 &
fi
FRONTEND_PID=$!
echo_ok "Frontend started (PID $FRONTEND_PID)"

# Wait for frontend
for i in $(seq 1 30); do
    if curl -s "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1; then
        echo_ok "Frontend ready"
        break
    fi
    sleep 2
done

# ---------- First-run setup ----------
FIRST_RUN_MARKER="$PROJECT_ROOT/.first-run-complete"
if [ ! -f "$FIRST_RUN_MARKER" ]; then
    echo -e "${CYAN}First-run setup: demo user${NC}"

    echo_step "Creating demo user..."
    TOKEN=$(curl -s -X POST "http://localhost:$BACKEND_PORT/api/auth/register" \
        -H "Content-Type: application/json" \
        -d '{"email":"demo@ecg.local","name":"Demo Clinician","password":"demo1234"}' \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('data',{}).get('token',''))" 2>/dev/null)

    if [ -z "$TOKEN" ]; then
        echo_step "Demo user may already exist — logging in..."
        TOKEN=$(curl -s -X POST "http://localhost:$BACKEND_PORT/api/auth/login" \
            -H "Content-Type: application/json" \
            -d '{"email":"demo@ecg.local","password":"demo1234"}' \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('data',{}).get('token',''))" 2>/dev/null)
    fi

    if [ -n "$TOKEN" ]; then
        echo_ok "Demo user ready"
    fi

    if [ $TRAIN_DEMO -eq 1 ] && [ -n "$TOKEN" ]; then
        echo_step "Training demo model (3 epochs)..."
        RUN_ID=$(curl -s -X POST "http://localhost:$BACKEND_PORT/api/training/start" \
            -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
            -d '{"model_name":"demo-model","epochs":3,"dataset_name":"synthetic"}' \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('data',{}).get('run_id',''))" 2>/dev/null)
        if [ -n "$RUN_ID" ]; then
            echo_step "Waiting for training to complete..."
            for i in $(seq 1 60); do
                sleep 5
                STATUS=$(curl -s -H "Authorization: Bearer $TOKEN" \
                    "http://localhost:$BACKEND_PORT/api/training/$RUN_ID" \
                    | python3 -c "import sys,json; print(json.load(sys.stdin).get('data',{}).get('status',''))" 2>/dev/null)
                if [ "$STATUS" = "completed" ]; then
                    echo_ok "Demo model trained!"
                    break
                elif [ "$STATUS" = "failed" ]; then
                    echo_err "Training failed"
                    break
                fi
            done
        fi
    fi

    touch "$FIRST_RUN_MARKER"
fi

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  All services are running!${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""
echo "  Frontend (Next.js):    http://localhost:$FRONTEND_PORT"
echo "  Backend  (FastAPI):    http://localhost:$BACKEND_PORT/health"
echo "  Backend  (API docs):   http://localhost:$BACKEND_PORT/docs"
echo "  WebSocket service:    ws://localhost:$WS_PORT"
echo ""
echo -e "  ${YELLOW}Login:    demo@ecg.local${NC}"
echo -e "  ${YELLOW}Password: demo1234${NC}"
echo ""
echo "  To stop everything:  kill $BACKEND_PID $WS_PID $FRONTEND_PID"
echo "  Or run:              ./stop-all.sh"
echo ""

read -p "Open browser now? [Y/n] " OPEN
if [ "$OPEN" != "n" ]; then
    if command -v open &>/dev/null; then open "http://localhost:$FRONTEND_PORT"
    elif command -v xdg-open &>/dev/null; then xdg-open "http://localhost:$FRONTEND_PORT"
    fi
fi
