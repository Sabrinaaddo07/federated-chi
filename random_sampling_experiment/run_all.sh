#!/usr/bin/env bash
# run_all.sh - Run all random sampling experiments in sequence.
#
# Runs K=2, 4, 6, 8, 10 (baseline) — one after another.
# Then generates the comparison plot.
#
# Usage:  bash run_all.sh

BASE_PID=$$
VALUES=(2 4 6 8 10)
NAMES=(
    "Pick 2 of 10 each round"
    "Pick 4 of 10 each round"
    "Pick 6 of 10 each round"
    "Pick 8 of 10 each round"
    "Pick 10 of 10 each round (baseline)"
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

for i in "${!VALUES[@]}"; do
    K="${VALUES[$i]}"
    NAME="${NAMES[$i]}"

    echo ""
    echo "========================================"
    echo "  Experiment $((i+1))/${#VALUES[@]}: $NAME"
    echo "========================================"
    echo ""

    # Start server in background
    echo "  Starting server (K=$K)..."
    python3 server.py --clients_per_round "$K" --num_clients 10 &
    SERVER_PID=$!
    sleep 3

    # Start all 10 clients
    echo "  Starting 10 clients..."
    for CID in $(seq 0 9); do
        python3 client.py --cid "$CID" --num_clients 10 &
        sleep 0.3
    done

    echo "  All clients started. Server will run for 40 rounds..."
    echo "  Waiting for server to finish..."

    wait "$SERVER_PID" 2>/dev/null || true

    echo ""
    echo "  Experiment $((i+1))/${#VALUES[@]} complete (results_sample_${K}.csv)"
    echo ""

    # Clean up clients before next run
    pkill -P "$BASE_PID" 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 3
done

echo ""
echo "========================================"
echo "  All experiments complete!"
echo "========================================"
echo ""
echo "  Results:"
for K in "${VALUES[@]}"; do
    if [ -f "results_sample_${K}.csv" ]; then
        echo "    results_sample_${K}.csv"
    fi
done
echo ""
echo "  Generating plot..."
python3 plot.py
echo ""
echo "  Open random_sampling.png to view."
