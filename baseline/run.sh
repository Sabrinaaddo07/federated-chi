#!/usr/bin/env bash
#
# run.sh — Baseline FL experiment
#
# Usage:
#   bash run.sh local              # Run all on one machine (testing)
#   bash run.sh chameleon <IP>     # Print instructions for Chameleon deployment
#   bash run.sh server             # Start just the server (for Chameleon server VM)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-local}"
SERVER_ADDR="${2:-10.0.0.1:8080}"

cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -P "$$" 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

if [ "$MODE" = "local" ]; then
    echo ""
    echo "========================================"
    echo "  BASELINE — Local (all on one machine)"
    echo "  10 clients, 40 rounds"
    echo "========================================"
    echo ""

    python3 server.py --server_address "127.0.0.1:8080" --num_clients 10 &
    SERVER_PID=$!
    sleep 3

    echo "  Starting 10 clients..."
    for CID in $(seq 0 9); do
        python3 client.py --cid "$CID" --num_clients 10 --server_address "127.0.0.1:8080" &
        sleep 0.3
    done

    echo "  Waiting for server to finish (40 rounds)..."
    wait "$SERVER_PID" 2>/dev/null || true

    echo ""
    echo "  Done! Results saved to results_baseline.csv"
    echo "  Run:  python3 plot.py  to generate plot"
    echo ""

elif [ "$MODE" = "chameleon" ]; then
    echo ""
    echo "========================================"
    echo "  BASELINE — Chameleon Deployment"
    echo "========================================"
    echo ""
    echo "  Server VM:"
    echo "    cd $(pwd)"
    echo "    python3 server.py --server_address 10.0.0.1:8080 --num_clients 10"
    echo ""
    echo "  Client VMs (run on each of 10 VMs):"
    for CID in $(seq 0 9); do
        echo "    VM $CID:  python3 client.py --cid $CID --num_clients 10 --server_address $SERVER_ADDR"
    done
    echo ""
    echo "  The server will wait until all 10 clients connect,"
    echo "  then run 40 rounds with 1.5s pauses between rounds."
    echo ""
    echo "  After completion, results_baseline.csv will be on the server VM."
    echo "  Run:  python3 plot.py  on the server VM (or copy CSV locally and run plot.py)."
    echo ""

elif [ "$MODE" = "server" ]; then
    echo "  Starting server on $SERVER_ADDR ..."
    python3 server.py --server_address "$SERVER_ADDR" --num_clients 10

else
    echo "Unknown mode: $MODE"
    echo "Usage: bash run.sh [local|chameleon|server] [server_address]"
    exit 1
fi
