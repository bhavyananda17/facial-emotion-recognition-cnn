"""Summarise multi-seed runs: paper vs clean accuracy for each seed, and the gap.

    python scripts/summarize_seeds.py ck

Reads results/<dataset>/{paper,clean}[_seed<N>]/run_info.json. The unsuffixed folders are the
headline seed-42 run. Writes results/<dataset>/multi_seed.md and prints it.
"""

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(folder):
    f = folder / "run_info.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return {"acc": r["metrics"]["test_accuracy"], "f1": r["metrics"]["macro_f1"], "epochs": r["epochs_run"],
            "seed": r["seed"]}


def main(dataset):
    base = ROOT / "results" / dataset
    seeds = {}
    for proto in ("paper", "clean"):
        for d in sorted(base.glob(f"{proto}*")):
            m = re.fullmatch(rf"{proto}(?:_seed(\d+))?", d.name)
            r = load(d) if m else None
            if r:
                seeds.setdefault(r["seed"], {})[proto] = r
    rows = [(sd, v["paper"], v["clean"]) for sd, v in sorted(seeds.items()) if {"paper", "clean"} <= v.keys()]
    n_test = None
    lines = [f"# {dataset}: paper vs clean protocol over {len(rows)} training seeds", "",
             "| Seed | Paper acc | Clean acc | Gap (paper - clean) | Paper epochs | Clean epochs |",
             "|---|---|---|---|---|---|"]
    gaps, pa, ca = [], [], []
    for sd, p, c in rows:
        gap = (p["acc"] - c["acc"]) * 100
        gaps.append(gap), pa.append(p["acc"] * 100), ca.append(c["acc"] * 100)
        tag = " (headline)" if sd == 42 else ""
        lines.append(f"| {sd}{tag} | {p['acc']:.1%} | {c['acc']:.1%} | {gap:+.1f} pts | {p['epochs']} | {c['epochs']} |")
    if len(rows) > 1:
        lines += ["", f"- Paper protocol: mean **{st.mean(pa):.1f}%**, std {st.stdev(pa):.1f}, range {min(pa):.1f} to {max(pa):.1f}",
                  f"- Clean protocol: mean **{st.mean(ca):.1f}%**, std {st.stdev(ca):.1f}, range {min(ca):.1f} to {max(ca):.1f}",
                  f"- Gap: mean **{st.mean(gaps):+.1f} pts**, std {st.stdev(gaps):.1f}, range {min(gaps):+.1f} to {max(gaps):+.1f}; "
                  f"paper protocol higher in {sum(g > 0 for g in gaps)} of {len(gaps)} seeds"]
    text = "\n".join(lines) + "\n"
    (base / "multi_seed.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ck")
