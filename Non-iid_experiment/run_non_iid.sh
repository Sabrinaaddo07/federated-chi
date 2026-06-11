#!/usr/bin/env bash
# run_non_iid.sh - Run all 3 non-IID experiments in sequence.
#
# Each experiment:
#   1. Starts the server in the background
#   2. Starts 8 clients in the background
#   3. Waits for the server to finish (40 rounds)
#   4. Kills all background processes
#   5. Waits a few seconds before the next experiment
#
# Usage:  bash run_non_iid.sh

set -e

BASE_PID=$$
EXPERIMENTS=("iid" "a" "b")
SCHEME_NAMES=(
    "IID baseline (balanced across all 10 classes)"
    "Non-IID A (1 digit class per client, classes 8-9 held out)"
    "Non-IID B (1 class/client, client 7 gets classes 7,8,9)"
)

cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    # Kill all child processes (servers + clients)
    pkill -P "$BASE_PID" 2>/dev/null || true
    # Also kill any lingering on port 8080
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 2
    echo "Cleanup done."
}

trap cleanup EXIT

for i in "${!EXPERIMENTS[@]}"; do
    SCHEME="${EXPERIMENTS[$i]}"
    NAME="${SCHEME_NAMES[$i]}"

    echo ""
    echo "========================================"
    echo "  Experiment $((i+1))/3: $NAME"
    echo "========================================"
    echo ""

    # Start server in background
    echo "  Starting server (scheme=$SCHEME)..."
    python3 server_non_iid.py --scheme "$SCHEME" &
    SERVER_PID=$!
    sleep 3

    # Start all 8 clients
    echo "  Starting 8 clients..."
    for CID in $(seq 0 7); do
        python3 client_non_iid.py --cid "$CID" --num_clients 8 --scheme "$SCHEME" &
        sleep 0.5
    done

    echo "  All clients started. Server will run for 40 rounds..."
    echo "  Waiting for server to finish..."

    # Wait for the server process to exit
    wait "$SERVER_PID" 2>/dev/null || true

    echo ""
    echo "  Experiment $((i+1))/3 complete (results_non_iid_${SCHEME}.csv)"
    echo ""

    # Clean up any leftover clients
    pkill -P "$BASE_PID" 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 3
done

echo ""
echo "========================================"
echo "  All 3 experiments complete!"
echo "========================================"
echo ""
echo "  Results:"
for SCHEME in "${EXPERIMENTS[@]}"; do
    if [ -f "results_non_iid_${SCHEME}.csv" ]; then
        echo "    results_non_iid_${SCHEME}.csv"
    fi
done
echo ""
echo "  To plot results, run:"
echo "    python3 plot_non_iid.py"
echo ""
