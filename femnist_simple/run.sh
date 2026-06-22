#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -P "$$" 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

for SCHEME in "iid" "non_iid"; do
    echo ""
    echo "========================================"
    echo "  Running FEMNIST: $SCHEME"
    echo "========================================"
    echo ""

    python3 server.py --scheme "$SCHEME" &
    SERVER_PID=$!
    sleep 3

    echo "  Starting 5 clients..."
    for CID in $(seq 0 4); do
        python3 client.py --cid "$CID" --num_clients 5 --scheme "$SCHEME" &
        sleep 0.3
    done

    echo "  Waiting for server to finish..."
    wait "$SERVER_PID" 2>/dev/null || true

    echo ""
    echo "  Done (results_femnist_simple_${SCHEME}.csv)"
    echo ""

    pkill -P "$$" 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 3
done

echo ""
echo "Both experiments complete!"
echo "Run:  python3 plot.py  to see results"
