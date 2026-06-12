#!/bin/bash
#
# run_all.sh — Automate model dropout experiments
#
# Experiment 1: Model dropout rate sweep (0.0, 0.25, 0.5, 0.75)
# Experiment 2: Client dropout comparison at equal communication budget
#
# Each experiment: kill old port, start server, start 8 clients, wait, clean up.

set -e

NUM_CLIENTS=8
ROUNDS=40
SERVER_PID=""
CLIENT_PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up processes..."
    for pid in "${CLIENT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    kill $(lsof -ti:8080) 2>/dev/null || true
    sleep 1
    echo "Done."
}
trap cleanup EXIT

wait_for_port_closed() {
    local port=8080
    for i in $(seq 1 30); do
        if ! nc -z 127.0.0.1 "$port" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    echo "  WARNING: Port $port still in use after 15s, continuing anyway..."
}

wait_for_server() {
    local port=8080
    for i in $(seq 1 30); do
        if nc -z 127.0.0.1 "$port" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    echo "ERROR: Server did not start on port $port"
    exit 1
}

run_experiment() {
    local mode=$1
    local dropout_rate=$2
    local clients_per_round=$3

    echo ""
    echo "============================================================"
    echo "  Running: mode=$mode dropout_rate=$dropout_rate cpr=$clients_per_round"
    echo "============================================================"
    echo ""

    # Kill anything on port 8080 and wait for it to close
    kill $(lsof -ti:8080) 2>/dev/null || true
    wait_for_port_closed

    # Start server
    python3 server.py \
        --mode "$mode" \
        --dropout_rate "$dropout_rate" \
        --clients_per_round "$clients_per_round" \
        --num_clients "$NUM_CLIENTS" &
    SERVER_PID=$!

    # Give server time to start binding, then wait for it
    sleep 3
    wait_for_server

    # Start all clients
    CLIENT_PIDS=()
    for cid in $(seq 0 $((NUM_CLIENTS - 1))); do
        python3 client.py \
            --cid "$cid" \
            --num_clients "$NUM_CLIENTS" \
            --dropout_rate "$dropout_rate" &
        CLIENT_PIDS+=($!)
    done

    # Wait for server to finish (40 rounds)
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""

    # Kill clients
    for pid in "${CLIENT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    CLIENT_PIDS=()

    # Wait for port to be released before next experiment
    sleep 5
    echo "Experiment complete."
}

# ── Experiment 1: Model dropout rate sweep ──
echo ""
echo "################################################################"
echo "  EXPERIMENT 1: Model Dropout Rate Sweep"
echo "################################################################"

for rate in 0.0 0.25 0.5 0.75; do
    run_experiment "model_dropout" "$rate" 8
done

# ── Experiment 2: Resource comparison ──
echo ""
echo "################################################################"
echo "  EXPERIMENT 2: Resource Comparison"
echo "################################################################"
echo ""
echo "  Model dropout 0.50 (8 clients x 50% = 4 full models/round)"
echo "  vs Client dropout K=4  (4 clients x 100% = 4 full models/round)"
echo ""

run_experiment "model_dropout" 0.50 8
run_experiment "client_dropout" 0.0 4

echo ""
echo "################################################################"
echo "  ALL EXPERIMENTS COMPLETE"
echo "################################################################"
echo ""
echo "Generating plots..."
python3 plot.py
echo ""
echo "Done!"
