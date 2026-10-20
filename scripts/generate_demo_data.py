#!/usr/bin/env python3
"""Generate a small synthetic network-flow dataset for a smoke test only."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic CICIDS-like CSV data for testing the code path.")
    parser.add_argument("--rows", type=int, default=20_000, help="Number of synthetic rows (default: 20000).")
    parser.add_argument("--output", default="data/demo/demo_network_traffic.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()
    if args.rows < 500:
        parser.error("--rows must be at least 500")

    rng = np.random.default_rng(args.seed)
    n = args.rows
    times = pd.date_range("2017-07-03 08:00:00", periods=n, freq="250ms")
    # Two regimes: attack traffic changes after the midpoint, creating a simple drift-like pattern.
    phase = np.arange(n) >= n // 2
    attack_prob = np.where(phase, 0.30, 0.12)
    is_attack = rng.random(n) < attack_prob
    labels = np.where(is_attack, np.where(rng.random(n) < 0.55, "DDoS", "PortScan"), "BENIGN")

    protocol = rng.choice([6, 17, 1], size=n, p=[0.72, 0.24, 0.04])
    benign_duration = rng.lognormal(mean=7.5, sigma=0.75, size=n)
    attack_duration = rng.lognormal(mean=6.1, sigma=1.05, size=n)
    # Later attacks increasingly resemble benign durations, so the online model has to adapt.
    duration = np.where(is_attack, attack_duration, benign_duration)
    late_attack = is_attack & phase
    duration[late_attack] = rng.lognormal(mean=7.0, sigma=1.1, size=int(late_attack.sum()))

    fwd_packets = np.where(is_attack, rng.poisson(22, n) + 1, rng.poisson(7, n) + 1)
    bwd_packets = np.where(is_attack, rng.poisson(2, n), rng.poisson(6, n))
    bytes_per_sec = np.where(is_attack, rng.lognormal(10.0, 1.2, n), rng.lognormal(8.0, 1.0, n))
    if phase.any():
        bytes_per_sec[late_attack] = rng.lognormal(8.8, 1.15, int(late_attack.sum()))
    flow_duration = np.maximum(duration, 1).astype(float)
    total_packets = fwd_packets + bwd_packets
    syn_flags = np.where(is_attack, rng.poisson(5, n), rng.poisson(1, n))
    dest_port = np.where(is_attack, rng.choice([80, 443, 4444, 8080], size=n), rng.choice([53, 80, 443, 123, 8080], size=n))
    frame = pd.DataFrame({
        "Timestamp": times.strftime("%d/%m/%Y %H:%M:%S.%f"),
        "Destination Port": dest_port,
        "Protocol": protocol,
        "Flow Duration": flow_duration,
        "Tot Fwd Pkts": fwd_packets,
        "Tot Bwd Pkts": bwd_packets,
        "TotLen Fwd Pkts": fwd_packets * rng.integers(60, 1200, n),
        "TotLen Bwd Pkts": bwd_packets * rng.integers(60, 1200, n),
        "Flow Bytes/s": bytes_per_sec,
        "Flow Packets/s": total_packets / (flow_duration / 1_000_000 + 0.1),
        "SYN Flag Cnt": syn_flags,
        "ACK Flag Cnt": np.where(is_attack, rng.poisson(1, n), rng.poisson(5, n)),
        "Label": labels,
    })
    # Split to imitate the multiple CSV files commonly supplied with CICIDS2017.
    out = (ROOT / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[: n // 2].to_csv(out, index=False)
    second_path = out.with_name(out.stem + "_part2" + out.suffix)
    frame.iloc[n // 2 :].to_csv(second_path, index=False)
    print(f"Created synthetic smoke-test files:\n  {out}\n  {second_path}")
    print("These are synthetic examples, not CICIDS2017 research results.")


if __name__ == "__main__":
    main()
