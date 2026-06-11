#!/usr/bin/env bash
# run_poison.sh - Run all 3 poisoning experiments in sequence.
#
# Each experiment:
#   1. Starts the server in the background
#   2. Starts 8 clients in the background (cid 7 is malicious)
#   3. Waits for the server to finish (40 rounds)
#   4. Kills all background processes
#   5. Waits a few seconds before the next experiment
#
# Usage:  bash run_poison.sh

set -e

BASE_PID=$$
EXPERIMENTS=("iid" "single" "multi")
SCHEME_NAMES=(
    "Exp 1: IID (balanced data, malicious poisons random labels on ~20 rounds)"
    "Exp 2: Non-IID single class (malicious has only 8s, others have 0-6)"
    "Exp 3: Non-IID multi class (malicious has 1,2,3, others have one each)"
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
    echo "  $NAME"
    echo "========================================"
    echo ""

    # Start server in background
    echo "  Starting server (scheme=$SCHEME)..."
    python3 server_poison.py --scheme "$SCHEME" &
    SERVER_PID=$!
    sleep 3

    # Start all 8 clients
    echo "  Starting 8 clients..."
    for CID in $(seq 0 7); do
        MALICIOUS_FLAG=""
        if [ "$CID" -eq 7 ]; then
            MALICIOUS_FLAG="--malicious"
        fi
        python3 client_poison.py --cid "$CID" --num_clients 8 --scheme "$SCHEME" $MALICIOUS_FLAG &
        sleep 0.5
    done

    echo "  All clients started. Server will run for 40 rounds..."
    echo "  Waiting for server to finish..."

    # Wait for the server process to exit
    wait "$SERVER_PID" 2>/dev/null || true

    echo ""
    echo "  Experiment complete (results_poison_${SCHEME}.csv)"
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
echo "  Results files:"
for SCHEME in "${EXPERIMENTS[@]}"; do
    if [ -f "results_poison_${SCHEME}.csv" ]; then
        echo "    results_poison_${SCHEME}.csv"
    fi
done
echo ""
echo "  To generate plots, run:"
echo "    python3 plot_poison.py"
echo ""
echo "  This creates:"
echo "    poison_experiment.png     — accuracy curves with poison bands"
echo "    poison_detectability.png  — malicious vs honest local accuracy"
echo "    poison_per_class.png      — per-class accuracy bar chart"
