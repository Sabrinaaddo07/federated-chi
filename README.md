
# Federated learning on Chameleon

By Sabrina Addo and Fraida Fund

In this repository, we expore federated learning on Chameleon using the Flower framework.
# federated-chi

Exploring federated learning with Flower + scikit-learn.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server (Terminal 1)
python3 server.py

# 3. Add clients (each in a new terminal)
python3 client.py --cid 0
python3 client.py --cid 1
python3 client.py --cid 2
# ... up to --cid 9

# 4. Training starts as soon as the first client connects.
#    Each round has a 10-second pause so you can add/remove clients.

# 5. Ctrl+C any client to disconnect it; server continues.
#    Ctrl+C the server to stop everything.
```

Data is split across up to 10 clients. Each client gets a unique non-overlapping slice. Only model weights are shared — raw data never leaves the client.

 (8 clients, clients gradually added in)
