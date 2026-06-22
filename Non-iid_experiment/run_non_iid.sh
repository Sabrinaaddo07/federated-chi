#!/usr/bin/env bash
# run_non_iid.sh - Run both experiments in sequence.
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
EXPERIMENTS=("iid" "non_iid")
SCHEME_NAMES=(
    "IID baseline (balanced across all 10 classes)"
    "Non-IID (1 dominant class + shared 8/9 per client, equal data)"
)

cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    pkill -P "$BASE_PID" 2>/dev/null || true
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
    echo "  Experiment $((i+1))/2: $NAME"
    echo "========================================"
    echo ""

    echo "  Starting server (scheme=$SCHEME)..."
    python3 server_non_iid.py --scheme "$SCHEME" &
    SERVER_PID=$!
    sleep 3

    echo "  Starting 8 clients..."
    for CID in $(seq 0 7); do
        python3 client_non_iid.py --cid "$CID" --num_clients 8 --scheme "$SCHEME" &
        sleep 0.5
    done

    echo "  All clients started. Server will run for 40 rounds..."
    echo "  Waiting for server to finish..."

    wait "$SERVER_PID" 2>/dev/null || true

    echo ""
    echo "  Experiment $((i+1))/2 complete (results_non_iid_${SCHEME}.csv)"
    echo ""

    pkill -P "$BASE_PID" 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 3
done

echo ""
echo "========================================"
echo "  Both experiments complete!"
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
