"""Live terminal dashboard for a running (or finished) training job.

    python scripts/watch_training.py results/rafdb/paper/train.log
    python scripts/watch_training.py results/rafdb/paper/train.log --once   # print once and exit
    python scripts/watch_training.py results/*/paper/train.log --once       # all datasets on one screen

It only reads the log file, so it is safe to start and stop at any time.
Uses the standard library only.
"""

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

EPOCH_LINE = re.compile(
    r"^\d+/\d+ - (\d+)s - \d+ms/step - accuracy: ([\d.]+) - loss: ([\d.]+) - "
    r"val_accuracy: ([\d.]+) - val_loss: ([\d.]+) - learning_rate: ([\de.+-]+)")
EPOCH_HEADER = re.compile(r"^Epoch (\d+)/(\d+)")
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, ORANGE, GREEN, RED = "\033[38;5;33m", "\033[38;5;208m", "\033[38;5;34m", "\033[38;5;160m"


def parse(log_path):
    rows, total, current = [], None, 0
    done = stopped = False
    final = None
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        if m := EPOCH_LINE.match(line):
            rows.append(dict(sec=int(m[1]), acc=float(m[2]), loss=float(m[3]), vacc=float(m[4]),
                             vloss=float(m[5]), lr=float(m[6])))
        elif m := EPOCH_HEADER.match(line):
            current, total = int(m[1]), int(m[2])
        elif "early stopping" in line:
            stopped = True
        elif line.startswith("TEST ACCURACY"):
            done, final = True, line.strip()
    return rows, total, current, done, stopped, final


def chart(rows, total, width, height=12, details=True):
    """One chart: train accuracy (o) and test/val accuracy (●) against epoch.

    The x axis always spans all `total` epochs, so the curves fill in from left to right.
    """
    lo = max(0.0, min(min(r["acc"] for r in rows), min(r["vacc"] for r in rows)) - 0.03)
    hi = min(1.0, max(max(r["acc"] for r in rows), max(r["vacc"] for r in rows)) + 0.03)
    span = (hi - lo) or 1
    grid = [[" "] * width for _ in range(height)]
    for key, mark in (("acc", f"{BLUE}o{RESET}"), ("vacc", f"{ORANGE}●{RESET}")):
        for i, r in enumerate(rows):
            x = round(i * (width - 1) / max(total - 1, 1))
            y = height - 1 - round((r[key] - lo) / span * (height - 1))
            grid[y][x] = mark
    lines = []
    for y, row in enumerate(grid):
        label = f"{hi - span * y / (height - 1):.2f}" if y in (0, height // 2, height - 1) else "    "
        lines.append(f"{DIM}{label:>5s} │{RESET}" + "".join(row))
    lines.append(f"{DIM}      └" + "─" * width + RESET)
    pad = " " * (width // 2 - 4)
    lines.append(f"{DIM}       1{pad}epoch{pad}{total}{RESET}")
    if details:  # the legend line is skipped in the compact multi-run view
        lines.append(f"       {BLUE}o{RESET} train accuracy    {ORANGE}●{RESET} test/val accuracy")
    return lines


NAMES = {"ck": "CK+", "rafdb": "RAF-DB", "fer2013": "FER2013"}


def label_for(log_path):
    """results/rafdb/paper/train.log -> 'RAF-DB · paper'."""
    parts = Path(log_path).parts
    if len(parts) >= 3 and parts[-3] in NAMES:
        return f"{NAMES[parts[-3]]} · {parts[-2]}"
    return str(log_path)


def render_compact(log_path, width):
    """A short view of one run (header + small accuracy chart) so several fit on one screen."""
    rows, total, current, done, stopped, final = parse(log_path)
    head = f"{BOLD}{label_for(log_path)}{RESET}"
    if not rows:
        return f"{head}  waiting for the first epoch..."
    n, last = len(rows), rows[-1]
    total = total or 100
    state = (f"{GREEN}FINISHED{RESET}" if done else f"{ORANGE}EARLY-STOPPED{RESET}" if stopped
             else f"{BLUE}TRAINING{RESET}")
    line = (f"{head}   {state} {n}/{total}   train acc {last['acc']:.3f}   test/val acc {BOLD}{last['vacc']:.3f}{RESET}"
            f"   {BLUE}o{RESET} train {ORANGE}●{RESET} test/val")
    if final:
        line += f"   {GREEN}{final.replace('TEST ACCURACY: ', 'final test: ')}{RESET}"
    return "\n".join([line] + chart(rows, total, min(width - 8, 90), height=6, details=False))


def render(log_path):
    rows, total, current, done, stopped, final = parse(log_path)
    width = min(shutil.get_terminal_size().columns, 100)
    out = [f"{BOLD}{log_path}{RESET}"]
    if not rows:
        out.append("Waiting for the first epoch to finish...")
        return "\n".join(out)

    n, last = len(rows), rows[-1]
    total = total or 100
    best = max(rows, key=lambda r: r["vacc"])
    best_loss_ep = min(range(n), key=lambda i: rows[i]["vloss"]) + 1
    avg = sum(r["sec"] for r in rows[-10:]) / len(rows[-10:])
    eta_min = (total - n) * avg / 60
    bar_w = width - 28
    filled = int(bar_w * n / total)
    state = (f"{GREEN}FINISHED{RESET}" if done else f"{ORANGE}EARLY-STOPPED{RESET}" if stopped
             else f"{BLUE}TRAINING{RESET}")
    out.append(f"{state}  epoch {BOLD}{n}/{total}{RESET}  [{BLUE}{'█' * filled}{DIM}{'░' * (bar_w - filled)}{RESET}] "
               f"{n / total:4.0%}")
    out.append(f"{DIM}{avg:.0f}s/epoch  ·  " + ("done" if done else f"ETA ≈ {eta_min:.0f} min (max, if no early stop)")
               + f"  ·  elapsed ≈ {sum(r['sec'] for r in rows) / 60:.0f} min{RESET}")
    out.append("")
    out.append(f"{'':16s}{'train':>10s}{'test/val':>10s}")
    out.append(f"{'accuracy':16s}{last['acc']:>10.3f}{BOLD}{last['vacc']:>10.3f}{RESET}")
    out.append(f"{'loss':16s}{last['loss']:>10.3f}{last['vloss']:>10.3f}")
    out.append(f"{'learning rate':16s}{last['lr']:>10.0e}")
    out.append("")
    out.extend(chart(rows, total, min(width - 8, 90)))
    out.append("")
    out.append(f"best val accuracy {BOLD}{best['vacc']:.3f}{RESET} (epoch {rows.index(best) + 1})  ·  "
               f"lowest val loss at epoch {best_loss_ep}")
    lr_drops = [i + 1 for i in range(1, n) if rows[i]["lr"] < rows[i - 1]["lr"]]
    if lr_drops:
        out.append(f"learning rate reduced at epoch(s): {', '.join(map(str, lr_drops))}")
    if final:
        out.append(f"\n{GREEN}{BOLD}{final}{RESET}")
    return "\n".join(out)


def render_all(logs):
    """One log: the full dashboard. Several logs: the compact view of each, stacked."""
    if len(logs) == 1:
        return render(logs[0])
    width = min(shutil.get_terminal_size().columns, 100)
    return "\n\n".join(render_compact(log, width) for log in logs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="path(s) to train.log; several logs are shown together, compactly")
    p.add_argument("--once", action="store_true", help="print once instead of refreshing")
    p.add_argument("--interval", type=float, default=5, help="seconds between refreshes")
    args = p.parse_args()
    if args.once or not sys.stdout.isatty():  # piped output: just print once
        print(render_all(args.logs))
        return
    # Alternate screen + cursor-home redraw: one view that updates in place, no stacking.
    sys.stdout.write("\033[?1049h\033[?25l")
    try:
        while True:
            frame = render_all(args.logs) + f"\n\n{DIM}refreshing every {args.interval:.0f}s, Ctrl+C to quit{RESET}"
            sys.stdout.write("\033[H" + "\033[K\n".join(frame.split("\n")) + "\033[K\033[J")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")


if __name__ == "__main__":
    main()
