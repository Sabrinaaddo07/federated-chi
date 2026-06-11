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

## Mid-experiment dropout experiment

Study how **sudden persistent client disconnections in the middle of training** affect model accuracy, convergence speed, stability, and data loss.

All 8 clients train together for **20 rounds**, then N clients are permanently dropped. The remaining clients continue for 20 more rounds.

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

### Run 2 — Drop 2 of 8 at round 20

```bash
python3 server.py --dropout 2 --dropout_round 20
# Then start all 8 clients in separate terminals
```

### Run 3 — Drop 4 of 8 at round 20

```bash
python3 server.py --dropout 4 --dropout_round 20
# Then start all 8 clients in separate terminals
```

### Run 4 — Drop 6 of 8 at round 20

```bash
python3 server.py --dropout 6 --dropout_round 20
# Then start all 8 clients in separate terminals
```

### How it works

- All 8 clients train rounds 1–19 (everyone contributes)
- At **round 20**, the server permanently drops N randomly selected clients
- The remaining 8-N clients train rounds 21–40
- Dropped clients still connect but the server ignores their updates
- Each run writes a CSV with round-by-round accuracy and data loss stats

### Plot

After all 4 runs, generate the comparison plot:

```bash
python3 plot.py
```

This outputs `dropout_experiment.png` and prints a metrics table:

| Metric | Dropout 0 | Dropout 2 | Dropout 4 | Dropout 6 |
|---|---|---|---|---|
| Final accuracy (r40) | | | | |
| Rounds to ≥80% | | | | |
| Accuracy variance (r30–40) | | | | |
| Gap vs baseline | — | | | |
| Data lost at dropout | 0% | | | |

### CSV format

```
round,dropout_count,dropout_round,global_accuracy,num_participated,num_available,participating_cids,dropped_data_share
1,2,20,0.3472,8,8,"[0,1,2,3,4,5,6,7]",0.0000
...
20,2,20,0.8917,6,8,"[0,2,3,5,6,7]",0.4490
...
```
