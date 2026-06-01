#!/usr/bin/env bash
# 启动 study-plan 前后端服务
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$SCRIPT_DIR/study-plan"
FRONTEND_DIR="$STUDY_DIR/frontend"

BACKEND_PORT="${BACKEND_PORT:-8765}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cleanup() {
  echo "Stopping services..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
}
trap cleanup EXIT

# Backend
echo "[backend] Starting on port $BACKEND_PORT..."
python "$STUDY_DIR/server.py" "$BACKEND_PORT" &
BACKEND_PID=$!

# Frontend
echo "[frontend] Starting on port $FRONTEND_PORT..."
cd "$FRONTEND_DIR"
if [ ! -d node_modules ]; then
  npm install
fi
npx vite --host 127.0.0.1 --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

echo ""
echo "✓ Backend:  http://127.0.0.1:$BACKEND_PORT"
echo "✓ Frontend: http://127.0.0.1:$FRONTEND_PORT"
echo "  Press Ctrl+C to stop"
wait
