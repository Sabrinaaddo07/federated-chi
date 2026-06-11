# Federated Learning on Chameleon

By Sabrina Addo and Fraida Fund

Exploring federated learning on Chameleon using Flower + scikit-learn.

## Quick start (basic demo)

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: start server
python3 server.py

# Terminals 2+: add clients (training starts immediately with 1 client)
python3 client.py --cid 0 --num_clients 8
python3 client.py --cid 1 --num_clients 8
# ... up to --cid 7

# Ctrl+C any client to disconnect; server continues.
# Ctrl+C server to stop.
```

Data is split across up to 8 clients. Each gets a unique non-overlapping slice of the digits dataset. Only model weights are shared — raw data never leaves the client.

## Dropout experiment

Study how persistent client disconnections affect model accuracy.

### Run 1 — Baseline (no dropout)

```bash
# Terminal 1
python3 server.py --dropout 0

# Terminals 2-9 (start all before round 1 begins)
python3 client.py --cid 0 --num_clients 8
python3 client.py --cid 1 --num_clients 8
...
python3 client.py --cid 7 --num_clients 8
```

### Run 2 — Drop 2 of 8 clients

```bash
python3 server.py --dropout 2
# Then start all 8 clients in separate terminals
```

### Run 3 — Drop 4 of 8 clients

```bash
python3 server.py --dropout 4
# Then start all 8 clients in separate terminals
```

### Run 4 — Drop 6 of 8 clients

```bash
python3 server.py --dropout 6
# Then start all 8 clients in separate terminals
```

### How it works

- Once all 8 clients connect, the server randomly selects N to **permanently drop** on round 1
- Dropped clients never participate in training again (they still connect, but the server ignores their updates)
- The remaining clients train every round as usual
- Each run writes a CSV file (`results_dropout_N.csv`) for plotting

### Expected results

| Dropout count | Final accuracy (round 40) | Rounds to reach 80% |
|---|---|---|
| 0 (8 of 8) | ~88% (baseline) | ~8-12 |
| 2 (6 of 8) | ~86% | ~15-20 |
| 4 (4 of 8) | ~82% | ~25+ |
| 6 (2 of 8) | ~75% | may not reach |

Higher dropout = slower convergence + lower final accuracy because less data contributes each round.

### CSV format

```
round,dropout_count,global_accuracy,num_participated,num_available,participating_cids
1,2,0.3472,6,8,"[0, 1, 2, 3, 4, 5]"
2,2,0.5028,6,8,"[0, 1, 2, 3, 4, 5]"
...
```
