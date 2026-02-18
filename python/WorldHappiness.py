# 0. IMPORTS

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# AUTO PATH DETECTION - works for anyone, no config needed

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

os.chdir(SCRIPT_DIR)

CSV_FILE = "WHR2023.csv"
if not os.path.exists(CSV_FILE):
    print(f"\nERROR: '{CSV_FILE}' not found in:\n  {SCRIPT_DIR}")
    print("Please place WHR2023.csv in the same folder as this script.")
    sys.exit(1)


# 1. LOAD DATA

df = pd.read_csv(CSV_FILE)

print("=" * 60)
print("WORLD HAPPINESS REPORT 2023 - EDA")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")


# 2. DATA CLEANING


# Rename columns to clean working names
df.rename(columns={
    "Country name"                               : "Country",
    "Ladder score"                               : "Happiness_Score",
    "Standard error of ladder score"             : "Std_Error",
    "upperwhisker"                               : "Upper_CI",
    "lowerwhisker"                               : "Lower_CI",
    "Logged GDP per capita"                      : "GDP_Per_Capita",
    "Social support"                             : "Social_Support",
    "Healthy life expectancy"                    : "Life_Expectancy",
    "Freedom to make life choices"               : "Freedom",
    "Generosity"                                 : "Generosity",
    "Perceptions of corruption"                  : "Corruption",
    "Ladder score in Dystopia"                   : "Dystopia_Score",
    "Explained by: Log GDP per capita"           : "Expl_GDP",
    "Explained by: Social support"               : "Expl_Social_Support",
    "Explained by: Healthy life expectancy"      : "Expl_Life_Expectancy",
    "Explained by: Freedom to make life choices" : "Expl_Freedom",
    "Explained by: Generosity"                   : "Expl_Generosity",
    "Explained by: Perceptions of corruption"    : "Expl_Corruption",
    "Dystopia + residual"                        : "Dystopia_Residual",
}, inplace=True)

# Drop rows with nulls in core columns
core_cols = ["Country", "Happiness_Score", "GDP_Per_Capita",
             "Social_Support", "Life_Expectancy", "Freedom",
             "Generosity", "Corruption"]

df.dropna(subset=core_cols, inplace=True)
df.drop_duplicates(subset="Country", inplace=True)
df.reset_index(drop=True, inplace=True)
df["Rank"] = df["Happiness_Score"].rank(ascending=False).astype(int)

print(f"Clean dataset: {df.shape[0]} countries ready for analysis.")


# 3. DESCRIPTIVE STATISTICS

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df[["Happiness_Score", "GDP_Per_Capita", "Social_Support",
          "Life_Expectancy", "Freedom", "Generosity", "Corruption"]].describe().round(3))


# 4. VISUALISATIONS

sns.set_theme(style="whitegrid", font_scale=1.1)
ACCENT   = "#1a6fc4"
BG_COLOR = "#f7f9fc"

plt.rcParams.update({
    "figure.facecolor"  : BG_COLOR,
    "axes.facecolor"    : BG_COLOR,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# -- PLOT 1: Top 10 & Bottom 10 Happiest Countries ---------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Happiest & Least Happy Countries (2023)",
             fontsize=16, fontweight="bold", y=1.01)

top10    = df.nlargest(10,  "Happiness_Score").sort_values("Happiness_Score")
bottom10 = df.nsmallest(10, "Happiness_Score").sort_values("Happiness_Score")

bars_top = axes[0].barh(top10["Country"], top10["Happiness_Score"],
                        color=sns.color_palette("Blues_r", 10))
axes[0].set_title("Top 10", fontweight="bold")
axes[0].set_xlabel("Happiness Score")
axes[0].set_xlim(0, 8.5)
for bar, val in zip(bars_top, top10["Happiness_Score"]):
    axes[0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)

bars_bot = axes[1].barh(bottom10["Country"], bottom10["Happiness_Score"],
                        color=sns.color_palette("Reds_r", 10))
axes[1].set_title("Bottom 10", fontweight="bold")
axes[1].set_xlabel("Happiness Score")
axes[1].set_xlim(0, 8.5)
for bar, val in zip(bars_bot, bottom10["Happiness_Score"]):
    axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("plot1_top_bottom_countries.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 1 saved.")

# -- PLOT 2: Distribution of Happiness Scores --------------------
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG_COLOR)

sns.histplot(df["Happiness_Score"], bins=20, kde=True,
             color=ACCENT, edgecolor="white", ax=ax)
ax.axvline(df["Happiness_Score"].mean(), color="#e05c2a", linestyle="--",
           linewidth=1.8, label=f'Mean: {df["Happiness_Score"].mean():.2f}')
ax.axvline(df["Happiness_Score"].median(), color="#2a9e4a", linestyle="--",
           linewidth=1.8, label=f'Median: {df["Happiness_Score"].median():.2f}')
ax.set_title("Distribution of Global Happiness Scores (2023)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Happiness Score")
ax.set_ylabel("Number of Countries")
ax.legend()

plt.tight_layout()
plt.savefig("plot2_score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 2 saved.")

# -- PLOT 3: Correlation Heatmap ---------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG_COLOR)

corr_cols   = ["Happiness_Score", "GDP_Per_Capita", "Social_Support",
               "Life_Expectancy", "Freedom", "Generosity", "Corruption"]
corr_matrix = df[corr_cols].corr()
mask_arr    = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask_arr, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            annot_kws={"size": 10}, ax=ax, square=True)
ax.set_title("Correlation Heatmap - Key Happiness Factors",
             fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("plot3_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 3 saved.")

# -- PLOT 4: GDP per Capita vs Happiness Score (scatter) ---------
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG_COLOR)

scatter = ax.scatter(df["GDP_Per_Capita"], df["Happiness_Score"],
                     c=df["Happiness_Score"], cmap="RdYlGn",
                     s=60, alpha=0.85, edgecolors="white", linewidths=0.4)

m, b   = np.polyfit(df["GDP_Per_Capita"], df["Happiness_Score"], 1)
x_line = np.linspace(df["GDP_Per_Capita"].min(), df["GDP_Per_Capita"].max(), 200)
ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5,
        linestyle="--", label="Trend line")

for _, row in df.nlargest(5, "Happiness_Score").iterrows():
    ax.annotate(row["Country"], (row["GDP_Per_Capita"], row["Happiness_Score"]),
                fontsize=7.5, textcoords="offset points", xytext=(5, 3))
for _, row in df.nsmallest(5, "Happiness_Score").iterrows():
    ax.annotate(row["Country"], (row["GDP_Per_Capita"], row["Happiness_Score"]),
                fontsize=7.5, textcoords="offset points", xytext=(5, -8))

plt.colorbar(scatter, ax=ax, label="Happiness Score")
ax.set_title("GDP per Capita vs Happiness Score (2023)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Logged GDP per Capita")
ax.set_ylabel("Happiness Score")
ax.legend()

plt.tight_layout()
plt.savefig("plot4_gdp_vs_happiness.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 4 saved.")

# -- PLOT 5: Factor Correlations with Happiness (bar chart) ------
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG_COLOR)

factors      = ["GDP_Per_Capita", "Social_Support", "Life_Expectancy",
                "Freedom", "Generosity", "Corruption"]
labels       = ["GDP per Capita", "Social Support", "Life Expectancy",
                "Freedom", "Generosity", "Corruption"]
correlations = [df["Happiness_Score"].corr(df[f]) for f in factors]
colors       = ["#2a9e4a" if c > 0 else "#e05c2a" for c in correlations]

bars = ax.barh(labels, correlations, color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Correlation of Each Factor with Happiness Score",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Pearson Correlation Coefficient (r)")
ax.set_xlim(-0.1, 1.0)

for bar, val in zip(bars, correlations):
    ax.text(val + 0.01 if val >= 0 else val - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=10)

plt.tight_layout()
plt.savefig("plot5_factor_correlations.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 5 saved.")

# -- PLOT 6: What Drives Happiness - Factor Breakdown (Top 20) ---
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)

factor_cols   = ["Expl_GDP", "Expl_Social_Support", "Expl_Life_Expectancy",
                 "Expl_Freedom", "Expl_Generosity", "Expl_Corruption",
                 "Dystopia_Residual"]
factor_labels = ["GDP per Capita", "Social Support", "Life Expectancy",
                 "Freedom", "Generosity", "Low Corruption", "Dystopia + Residual"]

top20          = df.nlargest(20, "Happiness_Score").sort_values("Happiness_Score", ascending=False)
factor_palette = sns.color_palette("tab10", len(factor_cols))
bottoms        = np.zeros(len(top20))

for col, label, color in zip(factor_cols, factor_labels, factor_palette):
    vals = top20[col].values
    ax.bar(top20["Country"], vals, bottom=bottoms,
           label=label, color=color, width=0.7,
           edgecolor="white", linewidth=0.3)
    bottoms += vals

ax.set_title("What Drives Happiness? - Factor Breakdown for Top 20 Countries",
             fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Happiness Score (explained)")
ax.tick_params(axis="x", rotation=45)
for label in ax.get_xticklabels():
    label.set_ha("right")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig("plot6_factor_breakdown.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 6 saved.")


# 5. KEY FINDINGS SUMMARY

print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

happiest   = df.loc[df["Happiness_Score"].idxmax(), "Country"]
unhappiest = df.loc[df["Happiness_Score"].idxmin(), "Country"]
max_score  = df["Happiness_Score"].max()
min_score  = df["Happiness_Score"].min()
mean_score = df["Happiness_Score"].mean()
corr_gdp   = df["Happiness_Score"].corr(df["GDP_Per_Capita"])
corr_soc   = df["Happiness_Score"].corr(df["Social_Support"])
corr_gen   = df["Happiness_Score"].corr(df["Generosity"])

print(f"""
  Happiest country    : {happiest} (score: {max_score:.2f})
  Least happy country : {unhappiest} (score: {min_score:.2f})
  Global mean score   : {mean_score:.2f}  |  Score range: {max_score - min_score:.2f}

  Strongest predictor : GDP per Capita (r = {corr_gdp:.2f})
  Social Support      : also highly correlated (r = {corr_soc:.2f})
  Generosity          : weakest correlation (r = {corr_gen:.2f})

  Wealth (GDP) and community (Social Support) account for
  the largest share of explained happiness across all nations.
""")

print(f"All 6 plots saved to: {SCRIPT_DIR}")
print("Analysis complete.")