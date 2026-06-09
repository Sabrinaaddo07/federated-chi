#!/bin/bash
#
# run.sh - One-click launcher for the Flower + scikit-learn FL demo.
#
# WHAT THIS DOES:
#   1. Starts the Flower server (the orchestrator).
#   2. Starts 2 Flower clients (the participants).
#   3. Each client trains on a DIFFERENT slice of the digits dataset.
#   4. After 5 rounds, prints the final accuracy.
#
# HOW IT WORKS:
#   The server and clients run as SEPARATE PROCESSES.
#   This simulates real federated learning where clients are
#   on different machines / phones / hospitals.
#
#   Communication happens over gRPC (port 8080).

set -e

echo ""
echo "======================================="
echo "  FEDERATED LEARNING DEMO"
echo "  Flower + scikit-learn (digits)"
echo "======================================="
echo ""
echo "  Server process:   python3 server.py"
echo "  Client 0 process: python3 client.py --cid 0"
echo "  Client 1 process: python3 client.py --cid 1"
echo ""
echo "  These run in parallel as separate processes."
echo "  Data is split: Client 0 gets rows 0-898, Client 1 gets 899-1796."
echo "  Only model weights travel between them. Raw data stays put."
echo ""

# Clean up any leftover processes from a previous run
pkill -f "server.py" 2>/dev/null || true
pkill -f "client.py" 2>/dev/null || true

# Start the server (listens on port 8080)
echo "[1/3] Starting Flower server..."
python3 server.py &
SERVER_PID=$!

# Give the server a moment to start listening
sleep 2

# Start the two clients (they'll connect to localhost:8080)
echo "[2/3] Starting Client 0..."
python3 client.py --cid 0 &
CLIENT0_PID=$!

echo "[3/3] Starting Client 1..."
python3 client.py --cid 1 &
CLIENT1_PID=$!

# Wait for clients to finish, then server
echo ""
echo "  Training in progress... (5 rounds, ~20 seconds)"
echo ""
wait $CLIENT0_PID $CLIENT1_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "  Demo finished."
echo ""
