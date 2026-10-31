"""Figures for RESULTS.md, drawn from the results/*.json files.

Palette: validated categorical slots 1-4 (blue, orange, aqua, yellow; adjacent
CVD dE >= 9.1). Aqua and yellow sit below 3:1 contrast on the light surface, so
every series is also direct-labelled and given its own marker, and every figure
has a table in RESULTS.md.
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
REF = "#8a8984"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MK = ["o", "s", "^", "D"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", color=INK, fontsize=9)


def fig_audit():
    p = os.path.join(RES, "e0_budget", "budget.json")
    if not os.path.exists(p):
        return
    res = json.load(open(p))
    ms = np.array(sorted(int(m) for m in res["audit_ceiling"]))
    ce = np.array([res["audit_ceiling"][str(m)] for m in ms])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ms, ce, color=C[0], marker="o", label="highest eps any attacker can certify")
    for claim in (4, 8):
        ax.axhline(claim, color=REF, lw=1)
        ax.annotate(f"claimed eps = {claim}", (ms[0], claim), xytext=(0, 4), textcoords="offset points",
                    color=INK2, fontsize=9)
    ax.axvline(128, color=C[1], lw=1.5)
    ax.annotate("plan: 128 canaries", (128, 0.4), xytext=(5, 0), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("canaries m (log scale)")
    ax.set_ylabel("95% lower bound on eps")
    ax.set_ylim(0, 9.5)
    ax.set_title("With 128 canaries the one-run audit cannot fail a claim of eps >= 3.74", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "audit_ceiling.png"), dpi=160)
    plt.close(fig)


def fig_filter_gain():
    p = os.path.join(RES, "e1_efficiency", "e1.json")
    if not os.path.exists(p):
        return
    rows = [r for r in json.load(open(p)) if r["eps"] == 4]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, tau in enumerate([0.05, 0.2, 0.5]):
        rs = sorted([r for r in rows if r["tau_cold"] == tau], key=lambda r: r["f_hot"])
        f = [r["f_hot"] for r in rs]
        g = [r["report"]["filter"]["0.03"]["gain_tokens_vs_fixed"] for r in rs]
        ax.plot(f, g, color=C[i], marker=MK[i], label=f"cold-token logit noise tau = {tau}")
        label_end(ax, f[-1], g[-1], f"tau = {tau}", C[i], dy=[8, 0, -8][i])
    ax.axhline(3, color=REF, lw=1)
    ax.annotate("G0 gate: 3x", (0.02, 3), xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("share of tokens where r matters (log scale)")
    ax.set_ylabel("filter gain over fixed b2 = 0.03")
    ax.set_xlim(0.015, 1.6)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("The filter clears 3x only if cold tokens barely move", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "filter_gain.png"), dpi=160)
    plt.close(fig)


def fig_sim():
    p = os.path.join(RES, "e4_simulator", "e4.json")
    if not os.path.exists(p):
        return
    res = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    ns = [512, 4096, 16384]
    mechs = [("m1_sampled", "M1 sampled token"), ("m1_full", "Gaussian, full vector"),
             ("m2_q0", "M2 DP-Fusion (q0 anchor)"), ("m2_unif", "uniform anchor")]
    for ax, world in zip(axes, ["state", "global"]):
        for i, (m, lab) in enumerate(mechs):
            ys, es = [], []
            for n in ns:
                r = [x for x in res if x["block"] == "A" and x["mech"] == m and x["world"]["kind"] == world
                     and x["n"] == n and x["eps"] == 4]
                ys.append(r[0]["H_mean"] if r else np.nan)
                es.append(r[0]["H_sd"] if r else np.nan)
            ax.errorbar(ns, ys, yerr=es, color=C[i], marker=MK[i], capsize=3, label=lab)
        npf = [[x for x in res if x["block"] == "A" and x["mech"] == "np_full" and x["world"]["kind"] == world
                and x["n"] == n][0]["H_mean"] for n in ns]
        ax.plot(ns, npf, color=REF, lw=1.5, ls="-", marker="", label="non-private, full teacher")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [f"{n:,}" for n in ns])
        ax.set_xlabel("records n")
        ax.set_title(f"{world} world, eps = 4", loc="left")
    axes[0].set_ylabel("headroom recovered H")
    axes[0].set_ylim(-0.3, 1.05)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Simulator: DP-Fusion with a q0 anchor recovers the most headroom", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "simulator_n.png"), dpi=160)
    plt.close(fig)


def fig_pilot():
    p = os.path.join(RES, "e5_pilot", "e5.json")
    if not os.path.exists(p):
        return
    res = json.load(open(p))
    ev = res["eval"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    eps_l = [1.0, 4.0, 8.0]
    for i, a in enumerate(["B1", "B2", "M1", "M2b"]):
        m, e = [], []
        for eps in eps_l:
            xs = [100 * r["exact"] for r in ev if r["arm"] == a and r["eps"] == eps]
            m.append(np.mean(xs))
            e.append(np.std(xs, ddof=1))
        ax.errorbar(eps_l, m, yerr=e, color=C[i], marker=MK[i], capsize=3, label=a)
        label_end(ax, eps_l[-1], m[-1], a, C[i])
    for a, lab in [("R0", "R0 base"), ("R2", "R2 OPSD (non-private)")]:
        v = 100 * np.mean([r["exact"] for r in ev if r["arm"] == a])
        ax.axhline(v, color=REF, lw=1)
        ax.annotate(lab, (1.0, v), xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_l, ["1", "4", "8"])
    ax.set_xlim(0.85, 11)
    ax.set_xlabel("eps (delta = 1e-6)")
    ax.set_ylabel("held-out exact match (%)")
    ax.set_title("Toy pilot, n = 4,096", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    rows = ev + res["dsweep"]
    ds = {0: 790, 240: 10390, 2032: 82070}
    for i, a in enumerate(["B1", "B2", "M1", "M2b"]):
        xs, ys = [], []
        for pad, d in ds.items():
            v = [100 * r["exact"] for r in rows if r["arm"] == a and r["eps"] == 4.0 and r["cfg"].get("p_pad", 0) == pad
                 and r["n"] == 4096]
            if v:
                xs.append(d)
                ys.append(np.mean(v))
        ax.plot(xs, ys, color=C[i], marker=MK[i], label=a)
        if xs:
            label_end(ax, xs[-1], ys[-1], a, C[i])
    for a, lab in [("R0", "R0 base")]:
        v = [100 * r["exact"] for r in rows if r["arm"] == a and r["cfg"].get("p_pad", 0) == 0 and r["n"] == 4096]
        ax.axhline(np.mean(v), color=REF, lw=1)
        ax.annotate(lab, (790, np.mean(v)), xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(500, 250000)
    ax.set_xlabel("trainable parameters d (log scale)")
    ax.set_title("Same task, more parameters, eps = 4", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pilot.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_audit()
    fig_filter_gain()
    fig_sim()
    fig_pilot()
    print(sorted(os.listdir(FIG)))
