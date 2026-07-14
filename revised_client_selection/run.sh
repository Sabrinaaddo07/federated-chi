#!/usr/bin/env bash
#
# run.sh — Revised Client Selection FL experiment
#
# Usage:
#   bash run.sh local               # Run all on one machine
#   bash run.sh chameleon <IP> <M>  # Print Chameleon instructions
#   bash run.sh server <addr> <M>   # Start just the server
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-local}"
SERVER_ADDR="${2:-10.0.0.1:8080}"
M="${3:-10}"

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
    echo "  REVISED CLIENT SELECTION — Local"
    echo "  10 clients, 75 rounds, M=$M"
    echo "========================================"
    echo ""

    echo "  Checking CIFAR-10 cache..."
    python3 -c "from common import _load_cifar10; _load_cifar10(); print('  CIFAR-10 ready')"

    python3 server.py --server_address "127.0.0.1:8080" --num_clients 10 \
        --num_rounds 75 --clients_per_round "$M" &
    SERVER_PID=$!
    sleep 3

    echo "  Starting 10 clients..."
    for CID in $(seq 0 9); do
        python3 client.py --cid "$CID" --num_clients 10 \
            --server_address "127.0.0.1:8080" &
    done

    wait "$SERVER_PID" 2>/dev/null || true

    echo ""
    echo "  Done! Results saved to results_revised_clientsel.csv"
    echo ""

elif [ "$MODE" = "chameleon" ]; then
    echo ""
    echo "========================================"
    echo "  REVISED CLIENT SELECTION — Chameleon"
    echo "========================================"
    echo ""
    echo "  Server VM:"
    echo "    cd $(pwd)"
    echo "    python3 server.py --server_address 10.0.0.1:8080 \\"
    echo "      --num_clients 10 --num_rounds 75 --clients_per_round $M"
    echo ""
    echo "  Client VMs (run on each of 10 VMs):"
    for CID in $(seq 0 9); do
        echo "    VM $CID:  python3 client.py --cid $CID --num_clients 10 \\"
        echo "      --server_address $SERVER_ADDR"
    done
    echo ""

elif [ "$MODE" = "server" ]; then
    echo "  Starting server on $SERVER_ADDR (M=$M) ..."
    python3 server.py --server_address "$SERVER_ADDR" --num_clients 10 \
        --num_rounds 75 --clients_per_round "$M"

else
    echo "Unknown mode: $MODE"
    echo "Usage: bash run.sh [local|chameleon|server] [server_address] [M]"
    exit 1
fi
