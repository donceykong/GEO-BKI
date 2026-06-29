"""Generate/refresh RUN.md for a Conv-BKI 9-class GT training run.

Idempotent: re-run it any time during or after training. It prefers the
authoritative train_history.json once that exists (written at run end); while
the run is live it parses the training log for per-epoch lines. A monitor loop
calls this every ~60s so RUN.md grows an epoch row as each epoch completes.

Usage:
    python write_run_md.py --meta meta.json --log run.log \
        --history train_history.json --out RUN.md
"""
from __future__ import annotations

import argparse
import json
import os
import re


EPOCH_VAL = re.compile(r"^\[epoch (\d+)\] VAL loss=([\d.nan]+) acc=([\d.nan]+) miou=([\d.nan]+)")
EPOCH_TRN = re.compile(r"^\[epoch (\d+)\] TRAIN loss=([\d.nan]+) acc=([\d.nan]+) miou=([\d.nan]+) iters=(\d+)")
FINAL_VAL = re.compile(r"^\[final\] VAL loss=([\d.nan]+) acc=([\d.nan]+) miou=([\d.nan]+)")
DONE = re.compile(r"^Done in ([\d.]+)s")


def parse_log(log_path):
    epochs = {}
    final_val = None
    done_s = None
    if not os.path.isfile(log_path):
        return epochs, final_val, done_s
    with open(log_path, errors="replace") as f:
        for line in f:
            line = line.strip()
            m = EPOCH_VAL.match(line)
            if m:
                e = int(m.group(1))
                epochs.setdefault(e, {})["val"] = {
                    "loss": m.group(2), "acc": m.group(3), "miou": m.group(4)}
                continue
            m = EPOCH_TRN.match(line)
            if m:
                e = int(m.group(1))
                epochs.setdefault(e, {})["train"] = {
                    "loss": m.group(2), "acc": m.group(3), "miou": m.group(4),
                    "iters": m.group(5)}
                continue
            m = FINAL_VAL.match(line)
            if m:
                final_val = {"loss": m.group(1), "acc": m.group(2), "miou": m.group(3)}
                continue
            m = DONE.match(line)
            if m:
                done_s = float(m.group(1))
    return epochs, final_val, done_s


def rows_from_history(hist):
    rows = []
    for ep in hist.get("epochs", []):
        e = ep["epoch"]
        v = ep.get("val", {})
        t = ep.get("train", {})
        rows.append({
            "epoch": e,
            "val_loss": v.get("loss"), "val_miou": v.get("miou"),
            "train_loss": t.get("loss"), "train_miou": t.get("miou"),
            "iters": t.get("iters"),
        })
    return rows


def fmt(x, nd=4):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--history", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    hist = None
    if args.history and os.path.isfile(args.history):
        try:
            with open(args.history) as f:
                hist = json.load(f)
        except json.JSONDecodeError:
            hist = None

    lines = []
    lines.append("# Conv-BKI 9-class GT retraining — kitti360_9class_gt\n")
    lines.append("GT-based retraining (project lead's design): the per-scan network "
                 "input is the **ground-truth one-hot** common label (not CENet "
                 "predictions); the target is the same GT. The per-class sparse kernel "
                 "learns the spatial extent per class that best reproduces GT at the "
                 "voxel level after multi-frame accumulation. Native **9-class common** "
                 "space — no SemKITTI 20-class projection.\n")

    lines.append("## Configuration\n")
    lines.append(f"- **Seed:** {meta.get('seed')}")
    lines.append(f"- **Grid:** {meta.get('grid')}")
    lines.append(f"- **Recipe:** num_frames={meta.get('num_frames')}, batch={meta.get('batch')}, "
                 f"epochs={meta.get('epochs')}, lr={meta.get('lr')} (Adam betas "
                 f"{meta.get('beta1')}/{meta.get('beta2')}), ExponentialLR γ={meta.get('decay')}, "
                 f"ell init={meta.get('ell_init')} (learnable), filter_size={meta.get('filter_size')}, "
                 f"per-class sparse kernel, NLL ignore_index=0")
    lines.append(f"- **Train split:** seqs {meta.get('train_seqs')} — "
                 f"{meta.get('train_scans')} scans")
    lines.append(f"- **Val split:** seqs {meta.get('val_seqs')} — "
                 f"{meta.get('val_scans')} scans (monitoring only; true held-out test is 0000/0009)")
    lines.append(f"- **Subsampling:** {meta.get('subsample')}")
    lines.append(f"- **Data root:** `{meta.get('data_root')}`")
    lines.append(f"- **Staging root:** `{meta.get('staging_root')}`")
    lines.append(f"- **Started:** {meta.get('started_at')}\n")

    lines.append("### Command\n")
    lines.append("```\n" + meta.get("command", "") + "\n```\n")

    lines.append("### Class weights (inverse log frequency from training GT)\n")
    lines.append("| common class | point count | NLL weight |")
    lines.append("|---|---|---|")
    names = meta.get("class_names", {})
    counts = meta.get("class_counts", [])
    nllw = meta.get("nll_weights", [])
    for i in range(len(counts)):
        nm = names.get(str(i), str(i))
        w = nllw[i] if i < len(nllw) else None
        note = " (ignored)" if i == 0 else ""
        lines.append(f"| {i} {nm}{note} | {counts[i]:,} | {fmt(w, 5)} |")
    lines.append("")

    lines.append("## Per-epoch history\n")
    lines.append("VAL is computed on the kernel entering that epoch (pre-train); "
                 "TRAIN is that epoch's mean. Both are GT-reproduction metrics, not "
                 "held-out.\n")
    lines.append("| epoch | val loss | val mIoU | train loss | train mIoU | train iters |")
    lines.append("|---|---|---|---|---|---|")

    if hist is not None:
        rows = rows_from_history(hist)
        for r in rows:
            lines.append(f"| {r['epoch']} | {fmt(r['val_loss'])} | {fmt(r['val_miou'])} | "
                         f"{fmt(r['train_loss'])} | {fmt(r['train_miou'])} | {r['iters'] or '—'} |")
        fv = hist.get("final_val")
        if fv:
            lines.append(f"| final | {fmt(fv.get('loss'))} | {fmt(fv.get('miou'))} | — | — | — |")
    else:
        epochs, final_val, done_s = parse_log(args.log)
        for e in sorted(epochs):
            v = epochs[e].get("val", {})
            t = epochs[e].get("train", {})
            lines.append(f"| {e} | {fmt(v.get('loss'))} | {fmt(v.get('miou'))} | "
                         f"{fmt(t.get('loss'))} | {fmt(t.get('miou'))} | {t.get('iters', '—')} |")
        if final_val:
            lines.append(f"| final | {fmt(final_val.get('loss'))} | {fmt(final_val.get('miou'))} | — | — | — |")

    lines.append("")

    # Final ell + total time (authoritative from history).
    if hist is not None:
        ell = hist.get("final_ell")
        if ell:
            lines.append("### Learned per-class `ell` (final)\n")
            cn = meta.get("class_names", {})
            lines.append("| class | ell |")
            lines.append("|---|---|")
            for i, e in enumerate(ell):
                lines.append(f"| {i} {cn.get(str(i), i)} | {fmt(e)} |")
            lines.append("")
        tot = hist.get("total_seconds")
        if tot:
            lines.append(f"**Total training time:** {tot:.0f}s ({tot/3600:.2f} h)\n")
        lines.append("_Status: training COMPLETE._\n")
    else:
        lines.append("_Status: training in progress — table refreshes per epoch._\n")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
