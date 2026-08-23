# -*- coding: utf-8 -*-
"""Figure 1 - SQLShield evaluation protocol. Drawn at final print size (6.4 in)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY, BLUEF   = "#1F4E78", "#DCE7F4"
AMBER, AMBERF = "#A05A00", "#FBEAD1"
GREY,  GREYF  = "#3F3F3F", "#ECECEC"
GREEN, GREENF = "#35682A", "#E3F0DB"
MUTED = "#6E6E6E"
DASH  = (0, (2.6, 1.9))

TS, BS, NS = 6.7, 5.35, 5.1          # title / body / note point sizes

fig, ax = plt.subplots(figsize=(6.4, 5.7), dpi=400)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, title, lines=None, ec=NAVY, fc=BLUEF, ls="solid",
        tsz=TS, bsz=BS, lsp=1.5):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.9", linewidth=0.9,
        edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2))
    if lines:
        ax.text(x, y+h/2-2.5, title, ha="center", va="center", zorder=3,
                fontsize=tsz, fontweight="bold", color=ec)
        ax.text(x, y+h/2-4.6, "\n".join(lines), ha="center", va="top", zorder=3,
                fontsize=bsz, color="#1A1A1A", linespacing=lsp)
    else:
        ax.text(x, y, title, ha="center", va="center", zorder=3,
                fontsize=tsz, fontweight="bold", color=ec)

def arr(p1, p2, color=NAVY, ls="solid", lw=0.95):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7,
        linewidth=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))

def note(x, y, s, size=NS, color=MUTED):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, color=color,
            style="italic", zorder=3, linespacing=1.45)

SX, SW = 33.0, 62.0        # spine  ->  2 .. 64
BX, BW = 83.0, 32.0        # branch -> 67 .. 99

ax.text(51, 97.5, "SQLShield evaluation protocol", ha="center", va="center",
        fontsize=8.4, fontweight="bold", color="#111111")

# ---- 1. two independent source lanes ------------------------------------
box(17.5, 90.0, 29, 8.6, "Source A",
    ["sajid576 / Modified_SQL_Dataset.csv", "text column: Query"])
box(48.5, 90.0, 29, 8.6, "Source B",
    ["syedsaqlainhussain / sqliv2.csv", "UTF-16 \u00b7 text column: Sentence"])

box(17.5, 77.3, 29, 8.8, "Strict cleaning",
    ["labels \u2208 {0,1} \u00b7 len > 2", "(text,label) dedup \u2192 30,766"],
    ec=AMBER, fc=AMBERF)
box(48.5, 77.3, 29, 8.8, "Strict cleaning",
    ["labels \u2208 {0,1} \u00b7 len > 2", "(text,label) dedup \u2192 33,537"],
    ec=AMBER, fc=AMBERF)
arr((17.5, 85.7), (17.5, 81.7)); arr((48.5, 85.7), (48.5, 81.7))
note(33, 83.6, "parsed separately", color=AMBER, size=5.3)

# ---- 2. merge + leakage control -----------------------------------------
box(SX, 64.5, SW, 9.4, "Merge + normalized-text leakage control",
    ["key = strip \u00b7 lowercase \u00b7 collapse whitespace",
     "drop contradictory groups \u00b7 exact cross-source dedup"],
    ec=AMBER, fc=AMBERF)
arr((17.5, 72.9), (25, 69.2)); arr((48.5, 72.9), (41, 69.2))

box(SX, 51.3, SW, 10.6, "Corrected corpus",
    ["56,621 rows \u00b7 34,490 benign / 22,131 SQLi",
     "45,915 normalized groups",
     "overlap audit: 54.73% of B, 59.67% of A shared"])
arr((SX, 59.8), (SX, 56.6))

# ---- 3. group-aware split ------------------------------------------------
box(SX, 38.3, SW, 9.4, "Group-aware split \u00b7 seed 42 \u00b7 80/10/10",
    ["train 45,288 \u00b7 validation 5,679 \u00b7 test 5,654",
     "zero normalized-text overlap asserted"],
    ec=AMBER, fc=AMBERF)
arr((SX, 46.0), (SX, 43.0))

# ---- 4. eight-model benchmark -------------------------------------------
box(SX, 24.5, SW, 12.0, "Fixed-test benchmark \u2014 8 models",
    ["transformers:  CodeBERT \u00b7 BERT-base",
     "word TF-IDF (1\u20132):  RF \u00b7 XGB \u00b7 LinearSVC \u00b7 LogReg",
     "char TF-IDF (3\u20135):  LinearSVC \u00b7 LogReg"], ec=GREY, fc=GREYF)
arr((SX, 33.6), (SX, 30.5), color=GREY)

# ---- 5. three fixed-test analyses ---------------------------------------
for cx, ttl, ls_ in [(12.5, "Stability",
                      ["5 seeds", "{7,21,42,84,126}", "mean \u00b1 SD"]),
                     (33.0, "Paired inference",
                      ["exact McNemar \u00d7 7", "Holm + Bonferroni", "95% Wilson CIs"]),
                     (53.5, "Error & cost audit",
                      ["per-sample predictions", "error complementarity",
                       "logged wall-clock"])]:
    box(cx, 9.0, 19.5, 11.8, ttl, ls_, ec=GREEN, fc=GREENF, tsz=6.4, bsz=5.1)
ax.plot([12.5, 53.5], [16.6, 16.6], color=GREEN, lw=0.95, zorder=1)
arr((SX, 18.5), (SX, 16.6), color=GREEN)
for cx in (12.5, 33.0, 53.5):
    arr((cx, 16.6), (cx, 14.9), color=GREEN)

# ---- branch A: cross-source audit (taps the CLEANED per-source data) -----
box(BX, 77.3, BW, 20.0, "Cross-source audit",
    ["trains on ONE cleaned source,",
     "tests on the other after removing",
     "every shared normalized-text row",
     "",
     "A\u2192B and B\u2192A \u00b7 90/10 group split",
     "validation-only threshold transfer",
     "benign source-fingerprint probe"],
    ec=NAVY, fc="#FFFFFF", ls=DASH, tsz=6.4, bsz=5.05, lsp=1.42)
ax.plot([63.0, 67.0], [77.3, 77.3], color=NAVY, lw=0.9, ls=DASH, zorder=1)
arr((65.4, 77.3), (67.0, 77.3), ls=DASH)

# ---- branch B: near-duplicate sensitivity (taps the CORPUS) -------------
box(BX, 51.3, BW, 19.5, "Near-duplicate sensitivity",
    ["independent re-split of the same",
     "56,621-row corrected corpus",
     "",
     "5-char shingles \u2192 MinHash-LSH \u2192",
     "verified Jaccard \u2265 \u03c4 \u2192 components",
     "\u03c4 \u2208 {0.9, 0.8, 0.7} \u00b7 StratGroupKFold",
     "CodeBERT seed 42 retrained"],
    ec=NAVY, fc="#FFFFFF", ls=DASH, tsz=6.4, bsz=5.05, lsp=1.42)
ax.plot([64.0, 67.0], [51.3, 51.3], color=NAVY, lw=0.9, ls=DASH, zorder=1)
arr((65.4, 51.3), (67.0, 51.3), ls=DASH)

note(BX, 37.0, "dashed = builds its own partitions;\nreported separately from the fixed test")

# ---- legend --------------------------------------------------------------
ly = 0.3
for i, (c, f, t) in enumerate([(AMBER, AMBERF, "leakage control"),
                               (NAVY, BLUEF, "data stage"),
                               (GREY, GREYF, "model bank"),
                               (GREEN, GREENF, "reported analysis")]):
    xx = 3.0 + i*17.5
    ax.add_patch(FancyBboxPatch((xx, ly), 2.4, 1.8,
        boxstyle="round,pad=0,rounding_size=0.35", linewidth=0.9,
        edgecolor=c, facecolor=f, zorder=2))
    ax.text(xx + 3.2, ly + 0.9, t, ha="left", va="center",
            fontsize=5.2, color="#333333", zorder=3)

plt.savefig("/home/claude/work/fig1_new.png", dpi=400, facecolor="white",
            bbox_inches="tight", pad_inches=0.05)
print("ok")
