#!/usr/bin/env bash
# run_poison.sh - Run all 4 poisoning experiments in sequence.
#
#   Exp 1: Baseline  — no malicious clients, all 8 train honestly
#   Exp 2: Uniform   — 3 malicious (CIDs 5,6,7) poison ~20 coordinated rounds
#   Exp 3: Early-only — same 3 malicious, poison ~10 rounds in 1-20 only
#   Exp 4: Late-only  — same 3 malicious, poison ~10 rounds in 21-40 only
#
# Usage:  bash run_poison.sh

set -e

BASE_PID=$$
EXPERIMENTS=("baseline" "uniform" "early_only" "late_only")
EXPERIMENT_FLAGS=(
    "--baseline"
    "--mode uniform"
    "--mode early_only"
    "--mode late_only"
)
NAMES=(
    "Exp 1: Baseline — no malicious (clean reference)"
    "Exp 2: Uniform — 3 attackers poison ~20 rounds across all 40"
    "Exp 3: Early-only — 3 attackers poison ~10 rounds in 1-20, then honest"
    "Exp 4: Late-only  — 3 attackers honest in 1-20, poison ~10 rounds in 21-40"
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
    LABEL="${EXPERIMENTS[$i]}"
    FLAGS="${EXPERIMENT_FLAGS[$i]}"
    NAME="${NAMES[$i]}"

    echo ""
    echo "========================================"
    echo "  $NAME"
    echo "========================================"
    echo ""

    # Start server in background
    echo "  Starting server ($FLAGS)..."
    python3 server_poison.py $FLAGS &
    SERVER_PID=$!
    sleep 3

    # Start all 8 clients
    echo "  Starting 8 clients..."
    for CID in $(seq 0 7); do
        MALICIOUS_FLAG=""
        if [ "$LABEL" != "baseline" ] && [ "$CID" -ge 5 ]; then
            MALICIOUS_FLAG="--malicious"
        fi
        python3 client_poison.py --cid "$CID" --num_clients 8 $MALICIOUS_FLAG &
        sleep 0.5
    done

    echo "  All clients started. Running for 40 rounds..."
    echo "  Waiting for server to finish..."

    wait "$SERVER_PID" 2>/dev/null || true

    echo ""
    echo "  Experiment complete (results_poison_${LABEL}.csv)"
    echo ""

    pkill -P "$BASE_PID" 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 3
done

echo ""
echo "========================================"
echo "  All 4 experiments complete!"
echo "========================================"
echo ""
echo "  Results files:"
for LABEL in "${EXPERIMENTS[@]}"; do
    if [ -f "results_poison_${LABEL}.csv" ]; then
        echo "    results_poison_${LABEL}.csv"
    fi
done
echo ""
echo "  To generate plots, run:"
echo "    python3 plot_poison.py"
echo ""
echo "  This creates:"
echo "    poison_experiment.png       — accuracy curves overlaid"
echo "    poison_detectability.png    — malicious vs honest local accuracy"
echo "    poison_per_class.png        — per-class accuracy bar chart"
