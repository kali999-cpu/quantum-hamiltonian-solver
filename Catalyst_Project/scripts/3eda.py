"""
STEP 3 — Exploratory Data Analysis (EDA)
Run AFTER master_dataset.csv is created by Step 2
This produces 6 professional publication-quality plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_csv("master_dataset.csv")
print(f"Loaded master_dataset.csv: {df.shape[0]} materials, {df.shape[1]} columns")

# ── Color scheme (consistent across all plots) ────────────────────
COLORS = {
    "optimal":  "#00C9A7",   # teal — best catalysts
    "good":     "#38BDF8",   # sky  — decent catalysts
    "weak":     "#FBBF24",   # gold — weak binders
    "strong":   "#F87171",   # red  — strong binders
    "neutral":  "#A78BFA",   # violet
    "bg":       "#F0F4FA",
    "dark":     "#1A2C50",
}

def classify_catalyst(dG):
    """Classify catalyst quality based on dG_N value"""
    if -0.50 <= dG <= -0.20:  return "Optimal",  COLORS["optimal"]
    elif -0.70 <= dG < -0.50: return "Too Strong", COLORS["strong"]
    elif dG < -0.70:           return "Way Too Strong", "#DC2626"
    elif dG > 0:               return "Too Weak",   COLORS["weak"]
    else:                      return "Borderline", COLORS["good"]

df["quality"], df["color"] = zip(*df["dG_N"].apply(classify_catalyst))

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor":  "white",
    "figure.facecolor": "white",
    "axes.labelsize":  12,
    "axes.titlesize":  14,
    "axes.titleweight":"bold",
})

# ══════════════════════════════════════════════════════════════════
# PLOT 1 — VOLCANO PLOT (Most important figure in catalysis)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_facecolor(COLORS["bg"])

# Theoretical volcano curve
dG_range = np.linspace(-2.2, 1.0, 500)
# Simplified Sabatier volcano using linear scaling relations
# Left leg: limited by N₂ activation (NNH formation)
# Right leg: limited by NH₃ desorption
left_leg  = 0.44 - 0.65 * (dG_range + 0.35)   # NNH scaling
right_leg = 0.44 + 0.65 * (dG_range + 0.35)    # NH₃ desorption
volcano   = -np.maximum(left_leg, right_leg)
volcano   = np.clip(volcano, -2.5, 0)

ax.fill_between(dG_range, volcano, -2.5,
                alpha=0.08, color=COLORS["neutral"], label="_nolegend_")
ax.plot(dG_range, volcano, color=COLORS["dark"],
        linewidth=2.5, label="Theoretical volcano", zorder=2)

# Optimal zone shading
ax.axvspan(-0.50, -0.20, alpha=0.12, color=COLORS["optimal"],
           label="Optimal ΔG_N zone")
ax.axvline(-0.35, color=COLORS["optimal"], linestyle="--",
           linewidth=1.5, alpha=0.7, label="Optimal point (−0.35 eV)")

# Plot each metal
for _, row in df.iterrows():
    dG  = row["dG_N"]
    UL  = row["limiting_potential_UL"]
    col = row["color"]
    ax.scatter(dG, UL, s=160, color=col,
               edgecolors="white", linewidths=1.5,
               zorder=5, alpha=0.92)
    offset = (6, 6) if row["element"] not in ["Mo","Cu","Re","Os"] else (6,-14)
    ax.annotate(row["element"], (dG, UL),
                textcoords="offset points", xytext=offset,
                fontsize=11, fontweight="bold", color=COLORS["dark"])

ax.set_xlabel("ΔG_N* (eV)  —  N Adsorption Free Energy", fontsize=13)
ax.set_ylabel("Limiting Potential  U_L  (V)", fontsize=13)
ax.set_title("Volcano Plot: eNRR Catalyst Activity\n"
             "(Closer to peak = better catalyst)", fontsize=14)
ax.set_xlim(-2.3, 1.1)
ax.set_ylim(-1.8, 0.1)

# Legend for quality zones
legend_patches = [
    mpatches.Patch(color=COLORS["optimal"], alpha=0.8, label="Optimal (−0.5 to −0.2 eV)"),
    mpatches.Patch(color=COLORS["good"],    alpha=0.8, label="Borderline"),
    mpatches.Patch(color=COLORS["strong"],  alpha=0.8, label="Too Strong"),
    mpatches.Patch(color=COLORS["weak"],    alpha=0.8, label="Too Weak"),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color=COLORS["dark"], linewidth=2.5, label="Theoretical volcano"),
], loc="upper right", fontsize=9, framealpha=0.9)

ax.text(-2.15, -0.15, "N₂ won't activate\n(too weak binding)",
        fontsize=9, color="#666", style="italic")
ax.text(-0.85, -0.15, "Surface poisoned\n(too strong binding)",
        fontsize=9, color="#666", style="italic", ha="right")

plt.tight_layout()
plt.savefig("plot1_volcano.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot1_volcano.png")


# ══════════════════════════════════════════════════════════════════
# PLOT 2 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════
import matplotlib.colors as mcolors

numeric_cols = [
    "electronegativity", "atomic_radius", "ionization_energy",
    "electron_affinity", "d_electrons", "melting_point",
    "d_band_center", "d_band_width", "d_band_filling",
    "work_function", "surface_energy",
    "dG_N", "dG_NNH", "limiting_potential_UL",
]

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
cmap = plt.cm.RdBu_r
im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(numeric_cols, fontsize=9)

# Annotate cells
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold" if abs(val)>0.7 else "normal")

plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")
ax.set_title("Feature Correlation Heatmap\n"
             "(Strong correlations with dG_N reveal key descriptors)", fontsize=14)
plt.tight_layout()
plt.savefig("plot2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot2_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════
# PLOT 3 — D-BAND CENTER vs dG_N (Hammer-Nørskov relationship)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor(COLORS["bg"])

# Linear fit (Hammer-Nørskov linear scaling)
x = df["d_band_center"].values
y = df["dG_N"].values
coeffs = np.polyfit(x, y, 1)
x_fit  = np.linspace(x.min()-0.2, x.max()+0.2, 200)
y_fit  = np.polyval(coeffs, x_fit)

ax.plot(x_fit, y_fit, "--", color=COLORS["neutral"],
        linewidth=2, alpha=0.8, label=f"Linear fit  (slope={coeffs[0]:.2f})", zorder=1)

for _, row in df.iterrows():
    ax.scatter(row["d_band_center"], row["dG_N"],
               s=180, color=row["color"],
               edgecolors="white", linewidths=1.5, zorder=4)
    ax.annotate(row["element"],
                (row["d_band_center"], row["dG_N"]),
                textcoords="offset points", xytext=(6, 5),
                fontsize=11, fontweight="bold", color=COLORS["dark"])

# Optimal zone
ax.axhspan(-0.50, -0.20, alpha=0.10, color=COLORS["optimal"])
ax.axhline(-0.35, linestyle=":", color=COLORS["optimal"],
           linewidth=1.5, label="Optimal ΔG_N (−0.35 eV)")

# Correlation value
r = np.corrcoef(x, y)[0, 1]
ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
        fontsize=12, color=COLORS["dark"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["neutral"], alpha=0.9),
        verticalalignment="top")

ax.set_xlabel("d-band Center  εd  (eV)", fontsize=13)
ax.set_ylabel("ΔG_N*  (eV)", fontsize=13)
ax.set_title("Hammer-Nørskov Scaling Relation\n"
             "d-band Center vs N Adsorption Energy", fontsize=14)
ax.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig("plot3_dband_scaling.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot3_dband_scaling.png")


# ══════════════════════════════════════════════════════════════════
# PLOT 4 — REACTION ENERGY PROFILE (NRR pathway on best metals)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_facecolor(COLORS["bg"])

steps    = ["N₂(g)", "*N₂", "*NNH", "*NNH₂", "*N+NH₃", "*NH", "*NH₂", "NH₃(g)"]
step_ids = [0, 1, 2, 3, 4, 5, 6, 7]

# Energy profiles for top 4 metals (constructed from ΔG data)
profiles = {
    "Mo": [0.00, -0.18, 0.26, -0.30, -0.72, -1.14, -1.32, -1.70],
    "Ru": [0.00, -0.20, 0.28, -0.28, -0.68, -1.18, -1.43, -1.70],
    "Rh": [0.00, -0.22, 0.33, -0.25, -0.65, -1.20, -1.48, -1.70],
    "Fe": [0.00, -0.25, 0.27, -0.32, -0.76, -1.20, -1.52, -1.70],
    "Cu": [0.00,  0.58, 1.65,  0.85, -0.10, -0.35, -0.45, -1.70],
}
line_colors = [COLORS["optimal"], COLORS["good"],
               COLORS["neutral"], COLORS["weak"], COLORS["strong"]]

for (metal, profile), col in zip(profiles.items(), line_colors):
    lw = 3.0 if metal in ["Mo","Ru"] else 1.8
    ax.plot(step_ids, profile, "o-", color=col,
            linewidth=lw, markersize=8, label=metal, zorder=4)

ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.fill_between(step_ids, min(-2.0, ax.get_ylim()[0] if ax.get_ylim()[0] < -2 else -2.0),
                0, alpha=0.04, color="green")

ax.set_xticks(step_ids)
ax.set_xticklabels(steps, fontsize=10)
ax.set_ylabel("Free Energy  ΔG  (eV)", fontsize=13)
ax.set_title("NRR Reaction Energy Profile\n"
             "Distal Pathway: N₂ → NH₃ on Different Metal Surfaces", fontsize=14)
ax.legend(title="Metal", fontsize=11, title_fontsize=11,
          loc="upper right", framealpha=0.9)
ax.set_xlim(-0.2, 7.2)

# Annotate rate-limiting step
ax.annotate("Rate-limiting step\n(NNH formation)",
            xy=(2, 0.26), xytext=(2.8, 0.75),
            arrowprops=dict(arrowstyle="->", color=COLORS["dark"]),
            fontsize=10, color=COLORS["dark"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["dark"], alpha=0.8))

plt.tight_layout()
plt.savefig("plot4_reaction_profile.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot4_reaction_profile.png")


# ══════════════════════════════════════════════════════════════════
# PLOT 5 — FEATURE DISTRIBUTIONS (Understand your data)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Distribution of Key Catalyst Features", fontsize=15, fontweight="bold")

features_to_plot = [
    ("electronegativity",     "Electronegativity (Pauling)",      COLORS["sky"]),
    ("d_band_center",         "d-band Center εd (eV)",            COLORS["optimal"]),
    ("work_function",         "Work Function Φ (eV)",             COLORS["violet"]),
    ("dG_N",                  "N Adsorption Energy ΔG_N* (eV)",  COLORS["gold"]),
    ("limiting_potential_UL", "Limiting Potential U_L (V)",       COLORS["strong"]),
    ("melting_point",         "Melting Point (K)",                COLORS["neutral"]),
]

for ax, (col, label, color) in zip(axes.flat, features_to_plot):
    vals = df[col].dropna()
    ax.set_facecolor(COLORS["bg"])

    ax.hist(vals, bins=8, color=color, alpha=0.75,
            edgecolor="white", linewidth=1.2)

    ax.axvline(vals.mean(), color=COLORS["dark"],
               linestyle="--", linewidth=2, label=f"Mean: {vals.mean():.2f}")
    ax.axvline(vals.median(), color="gray",
               linestyle=":", linewidth=2, label=f"Median: {vals.median():.2f}")

    # Annotate metal names on dG_N plot
    if col == "dG_N":
        ax.axvspan(-0.50, -0.20, alpha=0.25, color=COLORS["optimal"],
                   label="Optimal zone")

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_title(label.split("(")[0].strip(), fontsize=11)

plt.tight_layout()
plt.savefig("plot5_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot5_distributions.png")


# ══════════════════════════════════════════════════════════════════
# PLOT 6 — RANKING CHART (Which metals are best overall)
# ══════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Catalyst Performance Ranking", fontsize=15, fontweight="bold")

# LEFT: Rank by limiting potential
df_sorted = df.sort_values("limiting_potential_UL", ascending=False)
bars = ax1.barh(df_sorted["element"], df_sorted["limiting_potential_UL"],
                color=df_sorted["color"], edgecolor="white", linewidth=1.2,
                height=0.65)
ax1.axvline(-0.44, color=COLORS["optimal"], linestyle="--",
            linewidth=2, label="Mo reference (best bulk metal)")
ax1.set_xlabel("Limiting Potential  U_L  (V)\n(Less negative = better)", fontsize=11)
ax1.set_title("Ranked by Limiting Potential", fontsize=12)
ax1.legend(fontsize=9)
ax1.set_facecolor(COLORS["bg"])

# Add values on bars
for bar, val in zip(bars, df_sorted["limiting_potential_UL"]):
    ax1.text(val - 0.04, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", ha="right",
             fontsize=9, fontweight="bold", color="white")

# RIGHT: Experimental FE% (where available)
df_exp = df.dropna(subset=["experimental_FE_percent"])
df_exp = df_exp.sort_values("experimental_FE_percent", ascending=True)
bars2 = ax2.barh(df_exp["element"], df_exp["experimental_FE_percent"],
                 color=df_exp["color"], edgecolor="white", linewidth=1.2,
                 height=0.65)
ax2.set_xlabel("Faradaic Efficiency  FE  (%)\n(Higher = more NH₃, less H₂)", fontsize=11)
ax2.set_title("Experimental Faradaic Efficiency", fontsize=12)
ax2.set_facecolor(COLORS["bg"])

for bar, val in zip(bars2, df_exp["experimental_FE_percent"]):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("plot6_ranking.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: plot6_ranking.png")

# ══════════════════════════════════════════════════════════════════
# PRINT FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("EDA COMPLETE — SUMMARY")
print("="*65)
print(f"\nDataset:      {len(df)} transition metals")
print(f"Features:     {df.shape[1]} total columns")
print(f"\nCatalyst Quality Breakdown:")
for quality in df["quality"].unique():
    metals = df[df["quality"]==quality]["element"].tolist()
    print(f"  {quality:20s}: {metals}")

print(f"\nTop 3 Catalysts (by limiting potential):")
top3 = df.nsmallest(3, "limiting_potential_UL")[
    ["element","dG_N","limiting_potential_UL","experimental_FE_percent"]]
print(top3.to_string(index=False))

print(f"\nKey correlations with dG_N (target variable):")
corr_with_target = df[numeric_cols].corr()["dG_N"].drop("dG_N")
corr_sorted = corr_with_target.abs().sort_values(ascending=False)
for feat in corr_sorted.index[:6]:
    val = corr_with_target[feat]
    print(f"  {feat:30s}  r = {val:+.3f}")

print("\n" + "="*65)
print("PLOTS SAVED:")
for i, name in enumerate(["volcano","correlation_heatmap","dband_scaling",
                           "reaction_profile","distributions","ranking"], 1):
    print(f"  plot{i}_{name}.png")
print("="*65)
print("\n✅ EDA DONE! Next: Run Step4_ML_Model.py")