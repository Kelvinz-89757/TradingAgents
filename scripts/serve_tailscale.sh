#!/usr/bin/env bash
# Start the TradingAgents web UI and make it reachable from other devices on
# your tailnet (and nowhere else).
#
#   scripts/serve_tailscale.sh          # start
#   scripts/serve_tailscale.sh stop     # remove the `tailscale serve` mapping
#
# Two modes, picked automatically:
#   1. `tailscale serve` (HTTPS, https://<machine>.<tailnet>.ts.net/) when the
#      tailnet admin has enabled Serve. The server binds to localhost and
#      Tailscale is the only ingress.
#   2. Otherwise the server binds directly to this node's Tailscale IP
#      (http://100.x.y.z:PORT and http://<machine>.<tailnet>.ts.net:PORT).
#      The 100.x address is only routable inside the tailnet.
#
# Set TRADINGAGENTS_WEB_PORT to change the port (default 8501).
set -euo pipefail

PORT="${TRADINGAGENTS_WEB_PORT:-8501}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TS="${TAILSCALE_BIN:-tailscale}"
if ! command -v "$TS" >/dev/null 2>&1 && [ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]; then
  TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
fi
if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY=python; fi

if [ "${1:-}" = "stop" ]; then
  "$TS" serve --https=443 off 2>/dev/null || true
  echo "tailscale serve mapping removed"
  exit 0
fi

STATUS="$("$TS" status --json)"
HOST="$(printf '%s' "$STATUS" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
TSIP="$(printf '%s' "$STATUS" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["Self"]["TailscaleIPs"][0])')"

cd "$ROOT"
# `tailscale serve` blocks (polling) when Serve is not yet enabled on the
# tailnet, so give it a few seconds and fall back if it hasn't returned.
serve_ok=0
"$TS" serve --bg "http://127.0.0.1:${PORT}" >/tmp/ta-serve.out 2>&1 &
serve_pid=$!
for _ in 1 2 3 4 5 6; do
  if ! kill -0 "$serve_pid" 2>/dev/null; then
    wait "$serve_pid" && serve_ok=1
    break
  fi
  sleep 1
done
if [ "$serve_ok" != 1 ]; then kill "$serve_pid" 2>/dev/null || true; fi

if [ "$serve_ok" = 1 ]; then
  echo "Tailnet URL: https://${HOST}/"
  echo "Local URL:   http://127.0.0.1:${PORT}/"
  exec env TRADINGAGENTS_WEB_HOST=127.0.0.1 TRADINGAGENTS_WEB_PORT="$PORT" "$PY" -m web
else
  echo "tailscale serve is not enabled on this tailnet (enable it in the admin"
  echo "console to get HTTPS); binding to the Tailscale IP instead."
  echo "Tailnet URL: http://${HOST}:${PORT}/  (or http://${TSIP}:${PORT}/)"
  exec env TRADINGAGENTS_WEB_HOST="$TSIP" TRADINGAGENTS_WEB_PORT="$PORT" "$PY" -m web
fi
