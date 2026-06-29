#!/usr/bin/env bash
#
# run.sh — Structured Updates FL experiment
#
# Usage:
#   bash run.sh local <rate>         # Run all on one machine
#   bash run.sh chameleon <IP> <rate> # Print Chameleon instructions
#   bash run.sh server <addr> <rate>  # Start just the server
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-local}"
SERVER_ADDR="${2:-10.0.0.1:8080}"
RATE="${3:-1.0}"

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
    echo "  STRUCTURED UPDATES — Local"
    echo "  10 clients, 40 rounds, rate=$RATE"
    echo "========================================"
    echo ""

    python3 server.py --server_address "127.0.0.1:8080" --num_clients 10 \
        --update_rate "$RATE" &
    SERVER_PID=$!
    sleep 3

    echo "  Starting 10 clients..."
    for CID in $(seq 0 9); do
        python3 client.py --cid "$CID" --num_clients 10 \
            --update_rate "$RATE" --server_address "127.0.0.1:8080" &
    done

    wait "$SERVER_PID" 2>/dev/null || true

    echo ""
    echo "  Done! Results saved to results_structuredup.csv"
    echo ""

elif [ "$MODE" = "chameleon" ]; then
    echo ""
    echo "========================================"
    echo "  STRUCTURED UPDATES — Chameleon"
    echo "========================================"
    echo ""
    echo "  Server VM:"
    echo "    cd $(pwd)"
    echo "    python3 server.py --server_address 10.0.0.1:8080 \\"
    echo "      --num_clients 10 --update_rate $RATE"
    echo ""
    echo "  Client VMs (run on each of 10 VMs):"
    for CID in $(seq 0 9); do
        echo "    VM $CID:  python3 client.py --cid $CID --num_clients 10 \\"
        echo "      --update_rate $RATE --server_address $SERVER_ADDR"
    done
    echo ""

elif [ "$MODE" = "server" ]; then
    echo "  Starting server on $SERVER_ADDR (rate=$RATE) ..."
    python3 server.py --server_address "$SERVER_ADDR" --num_clients 10 \
        --update_rate "$RATE"

else
    echo "Unknown mode: $MODE"
    echo "Usage: bash run.sh [local|chameleon|server] [server_address] [rate]"
    exit 1
fi
