"""
run.py — Orchestrates all 25 experiments and generates the bar chart.

Runs every combination of:
  - Malicious %: 10, 20, 30, 40, 50
  - Run ID:      0, 1, 2, 3, 4

Each run: 10 clients, 40 rounds, 0.5s delay per round.

Usage:  python3 run.py
"""

import csv
import os
import signal
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "aggregated_results.csv")
PCTS = [10, 20, 30, 40, 50]
RUNS = 5
NUM_CLIENTS = 10
NUM_ROUNDS = 40


# ---------------------------------------------------------------------------
# Process management helpers
# ---------------------------------------------------------------------------

def kill_port(port=8080):
    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True, text=True, timeout=10,
    )
    pids = [p for p in result.stdout.strip().split("\n") if p]
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (OSError, ValueError):
            pass
    if pids:
        time.sleep(2)


def run_one_experiment(malicious_pct, run_id):
    """Start server, start clients, wait for completion, return final accuracy."""
    kill_port(8080)

    malicious_count = int(malicious_pct / 100 * NUM_CLIENTS)

    server_proc = subprocess.Popen(
        [sys.executable, "server.py",
         "--malicious_pct", str(malicious_pct),
         "--run_id", str(run_id)],
        cwd=HERE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(3)

    client_procs = []
    for cid in range(NUM_CLIENTS):
        cmd = [sys.executable, "client.py",
               "--cid", str(cid),
               "--num_clients", str(NUM_CLIENTS)]
        if cid < malicious_count:
            cmd.append("--malicious")
        proc = subprocess.Popen(
            cmd, cwd=HERE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client_procs.append(proc)

    print(f"    {malicious_pct}% malicious ({malicious_count}/{NUM_CLIENTS}), "
          f"run {run_id} — started server + {NUM_CLIENTS} clients")

    server_proc.wait()

    for p in client_procs:
        try:
            p.terminate()
        except Exception:
            pass

    csv_path = os.path.join(HERE, f"results_pct{malicious_pct}_run{run_id}.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        final_acc = float(rows[-1]["global_accuracy"]) if rows else 0.0
        print(f"    Finished — final accuracy: {final_acc:.4f}")
        return final_acc
    return 0.0


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    os.chdir(HERE)
    results = []

    for pct in PCTS:
        for run_id in range(RUNS):
            print(f"\n{'='*50}")
            print(f"  Malicious: {pct}%  |  Run {run_id + 1}/{RUNS}")
            print(f"{'='*50}")
            acc = run_one_experiment(pct, run_id)
            results.append({
                "malicious_pct": pct,
                "run_id": run_id,
                "final_accuracy": acc,
            })
            kill_port(8080)

    # Save aggregated results
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["malicious_pct", "run_id", "final_accuracy"])
        w.writeheader()
        w.writerows(results)
    print(f"\nAggregated results saved to {RESULTS_CSV}")

    # Generate bar chart
    plot_results(RESULTS_CSV)


def plot_results(csv_path):
    data = {pct: [] for pct in PCTS}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            data[int(row["malicious_pct"])].append(float(row["final_accuracy"]))

    means = []
    mins = []
    maxs = []
    for pct in PCTS:
        vals = data[pct]
        means.append(np.mean(vals))
        mins.append(np.min(vals))
        maxs.append(np.max(vals))

    lower_err = [m - mn for m, mn in zip(means, mins)]
    upper_err = [mx - m for m, mx in zip(means, maxs)]
    asymmetric_err = [lower_err, upper_err]

    x = np.arange(len(PCTS))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, means, yerr=asymmetric_err, capsize=8,
                  color=["#2e86ab", "#5cb85c", "#f6a623", "#d9534f", "#9b59b6"],
                  edgecolor="black", linewidth=1.2, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in PCTS], fontsize=12)
    ax.set_xlabel("Malicious Clients (%)", fontsize=13)
    ax.set_ylabel("Final Global Test Accuracy", fontsize=13)
    ax.set_title("Impact of Malicious Client Percentage on Federated Learning\n"
                 "(10 clients, 40 rounds, 5 runs each, random label-flipping)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend([bars], ["Accuracy range (min–max)"], fontsize=10)

    for i, (mean, mn, mx) in enumerate(zip(means, mins, maxs)):
        ax.annotate(f"{mean:.3f}", (x[i], mean), textcoords="offset points",
                    xytext=(0, -14 if mean > 0.5 else 8),
                    ha="center", fontsize=9, fontweight="bold",
                    color="black")

    plt.tight_layout()
    chart_path = os.path.join(HERE, "poison_percentage_results.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")

    print("\n" + "=" * 70)
    print(f"{'Malicious %':<15} {'Runs':<8} {'Mean Acc':<12} {'Min':<10} {'Max':<10} {'Std Dev':<10}")
    print("=" * 70)
    for pct in PCTS:
        vals = data[pct]
        print(f"{pct}%{'':<12} {len(vals):<8} {np.mean(vals):<12.4f} "
              f"{np.min(vals):<10.4f} {np.max(vals):<10.4f} {np.std(vals):<10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
