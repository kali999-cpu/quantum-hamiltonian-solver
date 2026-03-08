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

# ── Load data ─────────────────────────────────────
df = pd.read_csv("master_dataset.csv")
print(f"Loaded master_dataset.csv: {df.shape[0]} materials, {df.shape[1]} columns")

# ── Color scheme ──────────────────────────────────
COLORS = {
    "optimal": "#00C9A7",
    "good": "#38BDF8",
    "weak": "#FBBF24",
    "strong": "#F87171",
    "neutral": "#A78BFA",
    "bg": "#F0F4FA",
    "dark": "#1A2C50"
}

# ── Catalyst classification ───────────────────────
def classify_catalyst(dG):
    if -0.50 <= dG <= -0.20:
        return "Optimal", COLORS["optimal"]
    elif -0.70 <= dG < -0.50:
        return "Too Strong", COLORS["strong"]
    elif dG < -0.70:
        return "Way Too Strong", "#DC2626"
    elif dG > 0:
        return "Too Weak", COLORS["weak"]
    else:
        return "Borderline", COLORS["good"]

df["quality"], df["color"] = zip(*df["dG_N"].apply(classify_catalyst))

# ── Matplotlib style ──────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

# =================================================
# PLOT 1 — VOLCANO PLOT
# =================================================
fig, ax = plt.subplots(figsize=(10,6))
ax.set_facecolor(COLORS["bg"])

dG_range = np.linspace(-2.2,1.0,500)

left_leg = 0.44 - 0.65*(dG_range+0.35)
right_leg = 0.44 + 0.65*(dG_range+0.35)

volcano = -np.maximum(left_leg,right_leg)
volcano = np.clip(volcano,-2.5,0)

ax.plot(dG_range,volcano,color=COLORS["dark"],linewidth=2)

ax.axvspan(-0.50,-0.20,alpha=0.12,color=COLORS["optimal"])
ax.axvline(-0.35,color=COLORS["optimal"],linestyle="--")

for _,row in df.iterrows():
    ax.scatter(row["dG_N"],row["limiting_potential_UL"],
               color=row["color"],s=150,
               edgecolors="white")

    ax.text(row["dG_N"],row["limiting_potential_UL"],
            row["element"],fontsize=10)

ax.set_xlabel("ΔG_N* (eV)")
ax.set_ylabel("Limiting Potential U_L (V)")
ax.set_title("Volcano Plot for eNRR Catalysts")

plt.tight_layout()
plt.savefig("plot1_volcano.png",dpi=150)
plt.show()

print("Saved plot1_volcano.png")

# =================================================
# PLOT 2 — CORRELATION HEATMAP
# =================================================
numeric_cols = [
    "electronegativity","atomic_radius","ionization_energy",
    "electron_affinity","d_electrons","melting_point",
    "d_band_center","d_band_width","d_band_filling",
    "work_function","surface_energy",
    "dG_N","dG_NNH","limiting_potential_UL"
]

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10,8))

im = ax.imshow(corr,cmap="RdBu_r",vmin=-1,vmax=1)

ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))

ax.set_xticklabels(numeric_cols,rotation=45,ha="right")
ax.set_yticklabels(numeric_cols)

plt.colorbar(im,ax=ax)

ax.set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("plot2_correlation_heatmap.png",dpi=150)
plt.show()

print("Saved plot2_correlation_heatmap.png")

# =================================================
# PLOT 3 — d-band scaling relation
# =================================================
fig, ax = plt.subplots(figsize=(9,6))
ax.set_facecolor(COLORS["bg"])

x = df["d_band_center"]
y = df["dG_N"]

coeff = np.polyfit(x,y,1)

x_fit = np.linspace(x.min(),x.max(),100)
y_fit = np.polyval(coeff,x_fit)

ax.plot(x_fit,y_fit,"--",color=COLORS["neutral"])

for _,row in df.iterrows():
    ax.scatter(row["d_band_center"],row["dG_N"],
               color=row["color"],s=150)

    ax.text(row["d_band_center"],row["dG_N"],
            row["element"])

ax.set_xlabel("d-band center (eV)")
ax.set_ylabel("ΔG_N* (eV)")
ax.set_title("Hammer-Nørskov Scaling")

plt.tight_layout()
plt.savefig("plot3_dband_scaling.png",dpi=150)
plt.show()

print("Saved plot3_dband_scaling.png")

# =================================================
# PLOT 4 — Reaction Energy Profile
# =================================================
fig, ax = plt.subplots(figsize=(10,6))
ax.set_facecolor(COLORS["bg"])

steps=["N2","*N2","*NNH","*NNH2","*N+NH3","*NH","*NH2","NH3"]

profiles={
"Mo":[0,-0.18,0.26,-0.30,-0.72,-1.14,-1.32,-1.70],
"Ru":[0,-0.20,0.28,-0.28,-0.68,-1.18,-1.43,-1.70],
"Fe":[0,-0.25,0.27,-0.32,-0.76,-1.20,-1.52,-1.70],
"Cu":[0,0.58,1.65,0.85,-0.10,-0.35,-0.45,-1.70]
}

for metal,profile in profiles.items():
    ax.plot(range(len(profile)),profile,"o-",label=metal)

ax.set_xticks(range(len(steps)))
ax.set_xticklabels(steps)

ax.set_ylabel("Free Energy (eV)")
ax.set_title("NRR Reaction Energy Profile")

ax.legend()

plt.tight_layout()
plt.savefig("plot4_reaction_profile.png",dpi=150)
plt.show()

print("Saved plot4_reaction_profile.png")

# =================================================
# PLOT 5 — Feature Distributions
# =================================================
fig, axes = plt.subplots(2,3,figsize=(14,8))

features=[
("electronegativity","Electronegativity",COLORS["good"]),
("d_band_center","d-band center",COLORS["optimal"]),
("work_function","Work Function",COLORS["neutral"]),
("dG_N","N Adsorption Energy",COLORS["weak"]),
("limiting_potential_UL","Limiting Potential",COLORS["strong"]),
("melting_point","Melting Point",COLORS["neutral"])
]

for ax,(col,label,color) in zip(axes.flat,features):

    vals=df[col].dropna()

    ax.hist(vals,bins=8,color=color)

    ax.set_title(label)

plt.tight_layout()
plt.savefig("plot5_distributions.png",dpi=150)
plt.show()

print("Saved plot5_distributions.png")

# =================================================
# PLOT 6 — Ranking chart
# =================================================
fig, ax = plt.subplots(figsize=(8,6))

df_sorted=df.sort_values("limiting_potential_UL")

ax.barh(df_sorted["element"],df_sorted["limiting_potential_UL"],
        color=df_sorted["color"])

ax.set_xlabel("Limiting Potential (V)")
ax.set_title("Catalyst Ranking")

plt.tight_layout()
plt.savefig("plot6_ranking.png",dpi=150)
plt.show()

print("Saved plot6_ranking.png")

# =================================================
# SUMMARY
# =================================================
print("\nEDA COMPLETE")

top3=df.nsmallest(3,"limiting_potential_UL")

print("\nTop 3 Catalysts")
print(top3[["element","dG_N","limiting_potential_UL"]])