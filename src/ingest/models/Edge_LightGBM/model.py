import os
import time
import sys

import pandas as pd
import shap
import numpy as np
import lightgbm as lgb

from nba_api.stats.endpoints import BoxScoreTraditionalV2
from nba_api.stats.endpoints import ScoreboardV2, BoxScoreTraditionalV2
from nba_api.stats.static import teams

from sklearn.linear_model import Ridge
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

import matplotlib.pyplot as plt


# Functions
def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error between true and predicted values.
    
    Parameters
    ----------
    y_true : True target values
    y_pred : Predicted values
    
    Returns
    -------
    Mean absolute error
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if len(y_true) == 0:
        raise ValueError("y_true is empty, cannot compute MAE")
    
    return np.mean(np.abs(y_true - y_pred))

def bet_result(row):
    if row["bet_side"] == "NO BET":
        return np.nan

    # home bet
    if row["bet_side"] == "HOME":
        return row["home_margin"] > row["spread_signed"]

    # away bet
    if row["bet_side"] == "AWAY":
        return row["home_margin"] < row["spread_signed"]



INPUT_PATH = "data/processed/df_model_3.csv"

df_model = pd.read_csv(INPUT_PATH, parse_dates= ['date_x'])

TARGET = "home_margin"

FEATURES = [

    "spread_signed",
    "is_home_favorite",
    "home_margin_last5",
    "away_margin_last5",
    "home_b2b",
    "away_b2b",
    "pts_diff_last5",
    "margin_diff_last5",
    "rest_diff",
    "elo_diff",
    "home_elo_roll_2Y",
    "away_elo_roll_2Y",
    "rolling_margin_diff_5",
    "rolling_margin_diff_10",
    "rest_advantage",
    "rest_advantage_sq",
    "top8_points_diff",
    "travel_diff_3d",
    "travel_diff_7d",
    "travel_diff_10d",
    "travel_1d_z",
    "travel_3d_z",
    "travel_7d_z",
    "travel_10d_z",
    "fatigue_diff",

    "home_team_net_points_r5",
    "home_team_net_rebounds_r5",
    "home_team_net_assists_r5",
    "home_team_net_STL_r5",
    "home_team_net_BLK_r5",
    "home_team_net_turnovers_r5",
    "home_team_net_OREB_r5",
    "home_team_net_fgm_r5",
    "home_team_net_fga_r5",
    "home_team_net_three_pm_r5",
    "home_team_net_three_pa_r5",
    "home_team_net_ftm_r5",
    "home_team_net_fta_r5",

    "home_team_net_points_r10",
    "home_team_net_rebounds_r10",
    "home_team_net_assists_r10",
    "home_team_net_STL_r10",
    "home_team_net_BLK_r10",
    "home_team_net_turnovers_r10",
    "home_team_net_OREB_r10",
    "home_team_net_fgm_r10",
    "home_team_net_fga_r10",
    "home_team_net_three_pm_r10",
    "home_team_net_three_pa_r10",
    "home_team_net_ftm_r10",
    "home_team_net_fta_r10",

    "away_team_net_points_r5",
    "away_team_net_rebounds_r5",
    "away_team_net_assists_r5",
    "away_team_net_STL_r5",
    "away_team_net_BLK_r5",
    "away_team_net_turnovers_r5",
    "away_team_net_OREB_r5",
    "away_team_net_fgm_r5",
    "away_team_net_fga_r5",
    "away_team_net_three_pm_r5",
    "away_team_net_three_pa_r5",
    "away_team_net_ftm_r5",
    "away_team_net_fta_r5",

    "away_team_net_points_r10",
    "away_team_net_rebounds_r10",
    "away_team_net_assists_r10",
    "away_team_net_STL_r10",
    "away_team_net_BLK_r10",
    "away_team_net_turnovers_r10",
    "away_team_net_OREB_r10",
    "away_team_net_fgm_r10",
    "away_team_net_fga_r10",
    "away_team_net_three_pm_r10",
    "away_team_net_three_pa_r10",
    "away_team_net_ftm_r10",
    "away_team_net_fta_r10",
]
# optional: keep deterministic ordering
FEATURES = sorted(FEATURES)

df = df_model.copy()

# Ensure numeric
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.rename(columns={"date_x": "GAME_DATE"})
# Drop rows with missing values
#df = df.dropna(subset=FEATURES + [TARGET])

# Sort by time (CRITICAL)
df = df.sort_values("GAME_DATE").reset_index(drop=True)
# split_idx = int(len(df) * 0.8)

X_train = df.loc[df["season"] < 2025, FEATURES]
y_train = df.loc[df["season"] < 2025, TARGET]

X_test = df.loc[df["season"] == 2025, FEATURES]
y_test = df.loc[df["season"] == 2025, TARGET]


model = lgb.LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=4,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

residuals = y_train - model.predict(X_train)
sigma_hat = residuals.std()

print("Estimated sigma:", sigma_hat)

print(residuals.mean(), residuals.std())

preds = model.predict(X_test)
print("MAE:", mae(y_test, preds))

imp = pd.Series(model.feature_importances_, index=FEATURES)
print(imp.sort_values(ascending=False).head(20))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train.sample(2000, random_state=42))

shap.summary_plot(shap_values, X_train.sample(2000, random_state=42))

df_test = df.loc[df["season"] == 2025].copy()

df_test["model_margin"] = preds
df_test["edge"] = df_test["model_margin"] - df_test["spread_signed"]

from math import erf, sqrt

def normal_cdf(x, mu=0, sigma=sigma_hat):
    return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))


####start
# Standardize edge using residual std
df_test["edge_std"] = (df_test["model_margin"] - df_test["spread_signed"]) / sigma_hat

# Compute win probability using standard normal CDF
from scipy.stats import norm
df_test["win_prob_home"] = norm.cdf(df_test["edge_std"])  # home win probability

# EV calculation
PAYOUT = 0.909
df_test["ev_home"] = df_test["win_prob_home"] * PAYOUT - (1 - df_test["win_prob_home"])
df_test["ev_away"] = (1 - df_test["win_prob_home"]) * PAYOUT - df_test["win_prob_home"]

# Set bet side based on EV threshold
EV_THRESHOLD = 0.02  # 2% edge
df_test["bet_side"] = np.where(
    df_test["ev_home"] > EV_THRESHOLD, "HOME",
    np.where(df_test["ev_away"] > EV_THRESHOLD, "AWAY", "NO BET")
)

# Compute P&L
df_test["bet_ev"] = np.where(
    df_test["bet_side"] == "HOME", df_test["ev_home"],
    np.where(df_test["bet_side"] == "AWAY", df_test["ev_away"], 0)
)
# df_test["bet_win"] = df_test.apply(bet_result, axis=1)


# 1️⃣ Home margin needed to cover
df_test["home_margin_needed_to_cover"] = np.where(
    df_test["is_home_favorite"] == 1,
    df_test["spread"],         # home is favorite → needs to beat spread
    -1 * df_test["spread"]     # home is underdog → needs to beat -spread
)

# 2️⃣ Away margin needed to cover
df_test["away_margin_needed_to_cover"] = -1 * df_test["home_margin_needed_to_cover"]

# 3️⃣ Did home cover?
df_test["did_home_cover"] = df_test["home_margin"] > df_test["home_margin_needed_to_cover"]

# 4️⃣ Did away cover?
df_test["did_away_cover"] = df_test["away_margin"] > df_test["away_margin_needed_to_cover"]

# 5️⃣ Which team covered?
df_test["team_that_covered"] = np.where(
    df_test["did_home_cover"],
    "HOME",
    "AWAY"
)

# 6️⃣ Bet win based on bet_side
df_test["bet_win"] = (df_test["bet_side"] == df_test["team_that_covered"]).astype(int)
#####


# df_test["bet_win"] = (
#     ((df_test["home_margin"] > df_test["spread_signed"]) & (df_test["bet_side"] == "HOME")) |
#     ((df_test["home_margin"] < df_test["spread_signed"]) & (df_test["bet_side"] == "AWAY"))
# ).astype(int)

####
# -----------------------
# Kelly bet sizing
# -----------------------
PAYOUT = 0.909          # decimal odds -1 for -110 bet
FRACTIONAL_KELLY = 0.5  # use 50% of full Kelly to reduce variance (optional)

def kelly_fraction(p, b):
    """
    Compute Kelly fraction for even/decimal odds.

    Parameters
    ----------
    p : win probability
    b : payout (decimal - 1)
    
    Returns
    -------
    fraction of bankroll to bet (0 to 1)
    """
    f = (p * (b + 1) - 1) / b
    return np.clip(f, 0, 1)  # no negative bets, cap at 1

# Initialize column
df_test["kelly_frac"] = 0.0

# HOME bets
home_mask = df_test["bet_side"] == "HOME"
df_test.loc[home_mask, "kelly_frac"] = kelly_fraction(
    df_test.loc[home_mask, "win_prob_home"],
    PAYOUT
)

# AWAY bets
away_mask = df_test["bet_side"] == "AWAY"
df_test.loc[away_mask, "kelly_frac"] = kelly_fraction(
    1 - df_test.loc[away_mask, "win_prob_home"],  # probability away wins
    PAYOUT
)

# Apply fractional Kelly if desired
df_test["kelly_frac"] *= FRACTIONAL_KELLY

# -----------------------
# Compute Kelly-scaled P&L
# -----------------------
df_test["pnl_kelly"] = 0.0

# Wins
df_test.loc[df_test["bet_win"] == 1, "pnl_kelly"] = df_test.loc[df_test["bet_win"] == 1, "kelly_frac"] * PAYOUT

# Losses
df_test.loc[df_test["bet_win"] == 0, "pnl_kelly"] = -df_test.loc[df_test["bet_win"] == 0, "kelly_frac"]

# No bet
df_test.loc[df_test["bet_side"] == "NO BET", "pnl_kelly"] = 0

# -----------------------
# Summarize Kelly bets
# -----------------------
bets = df_test[df_test["bet_side"] != "NO BET"].copy()
bets["cum_pnl_kelly"] = bets["pnl_kelly"].cumsum()

summary_kelly = {
    "num_bets": len(bets),
    "win_rate": bets["bet_win"].mean(),
    "total_pnl": bets["pnl_kelly"].sum(),
    "roi": bets["pnl_kelly"].sum() / len(bets),
    "avg_ev": bets["bet_ev"].mean(),
}

print("Kelly Betting Summary:", summary_kelly)

# -----------------------
# Plot cumulative P&L
# -----------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(bets["GAME_DATE"], bets["cum_pnl_kelly"])
plt.title("Cumulative P&L – LightGBM Spread Model (Kelly Sizing)")
plt.xlabel("Date")
plt.ylabel("Units")
plt.grid(True)
plt.show()



bets = df_test[df_test["bet_side"] != "NO BET"]


summary = {
    "num_bets": len(bets),
    "win_rate": bets["bet_win"].mean(),
    "total_pnl": bets["pnl_kelly"].sum(),
    "roi": bets["pnl_kelly"].sum() / len(bets),
    "avg_ev": bets["bet_ev"].mean(),
}

print(summary)

bets = bets.sort_values("GAME_DATE")
bets["cum_pnl"] = bets["pnl_kelly"].cumsum()


plt.figure(figsize=(10,5))
plt.plot(bets["GAME_DATE"], bets["cum_pnl"])
plt.title("Cumulative P&L – LightGBM Spread Model")
plt.xlabel("Date")
plt.ylabel("Units")
plt.grid(True)
plt.show()

# -----------------------
# Edge buckets (Kelly-weighted)
# -----------------------
bets["edge_bucket"] = pd.cut(
    bets["edge"],
    bins=[-10, -4, -2, 0, 2, 4, 10]
)

edge_summary = bets.groupby("edge_bucket", observed=False).agg(
    num_bets=("pnl_kelly", "count"),
    total_pnl=("pnl_kelly", "sum"),
    roi=("pnl_kelly", lambda x: x.sum() / x.count()),  # average units risked per bet
    win_rate=("bet_win", "mean")
).reset_index()

print("Edge Bucket Summary (Kelly-Weighted):")
print(edge_summary)

# -----------------------
# Probability buckets (Kelly-weighted)
# -----------------------
bets["prob_bucket"] = pd.cut(
    bets["win_prob_home"],
    bins=np.linspace(0, 1, 11)
)

prob_summary = bets.groupby("prob_bucket", observed=False).agg(
    num_bets=("pnl_kelly", "count"),
    total_pnl=("pnl_kelly", "sum"),
    roi=("pnl_kelly", lambda x: x.sum() / x.count()),
    win_rate=("bet_win", "mean")
).reset_index()

print("\nProbability Bucket Summary (Kelly-Weighted):")
print(prob_summary)


os.makedirs("outputs", exist_ok=True)


PAYOUT = 0.909           # decimal odds -1 for -110
FRACTIONAL_KELLY = 0.1   # fraction of full Kelly to reduce volatility
EV_THRESHOLD = 0.1      # minimum expected value to place a bet

# -----------------------
# Read data
# -----------------------
df_test = bets.copy()

sigma_hat = df_test["edge"].std()

# -----------------------
# Only calibrate probabilities where edge exists
mask = df_test["bet_side"] != "NO BET"  # use only non-NaN bets for calibration
ir = IsotonicRegression(out_of_bounds='clip')
# Fake target: home covered or not
df_test["home_cover"] = df_test["home_margin"] > df_test["spread_signed"]
df_test["win_prob_home_calibrated"] = ir.fit_transform(df_test.loc[mask, "win_prob_home"],
                                                      df_test.loc[mask, "home_cover"].astype(int))
# Use calibrated probabilities
df_test["win_prob_home"] = df_test["win_prob_home_calibrated"]

# -----------------------
# Define margin coverage
# -----------------------
df_test["home_margin_needed_to_cover"] = np.where(
    df_test["is_home_favorite"] == 1,
    df_test["spread"],
    -1 * df_test["spread"]
)
df_test["away_margin_needed_to_cover"] = -1 * df_test["home_margin_needed_to_cover"]
df_test["did_home_cover"] = df_test["home_margin"] > df_test["home_margin_needed_to_cover"]
df_test["did_away_cover"] = df_test["away_margin"] > df_test["away_margin_needed_to_cover"]
df_test["team_that_covered"] = np.where(df_test["did_home_cover"], "HOME", "AWAY")
df_test["bet_win"] = (df_test["bet_side"] == df_test["team_that_covered"]).astype(int)

# -----------------------
# Compute EVs for home/away
# -----------------------
df_test["ev_home"] = df_test["win_prob_home"] * PAYOUT - (1 - df_test["win_prob_home"])
df_test["ev_away"] = (1 - df_test["win_prob_home"]) * PAYOUT - df_test["win_prob_home"]

# Apply EV threshold to determine bet side
df_test["bet_side"] = np.where(
    df_test["ev_home"] > EV_THRESHOLD, "HOME",
    np.where(df_test["ev_away"] > EV_THRESHOLD, "AWAY", "NO BET")
)

# -----------------------
# Kelly sizing
# -----------------------
def kelly_fraction(p, b):
    f = (p * (b + 1) - 1) / b
    return np.clip(f, 0, 1)

df_test["kelly_frac"] = 0.0
home_mask = df_test["bet_side"] == "HOME"
away_mask = df_test["bet_side"] == "AWAY"

df_test.loc[home_mask, "kelly_frac"] = kelly_fraction(df_test.loc[home_mask, "win_prob_home"], PAYOUT)
df_test.loc[away_mask, "kelly_frac"] = kelly_fraction(1 - df_test.loc[away_mask, "win_prob_home"], PAYOUT)
df_test["kelly_frac"] *= FRACTIONAL_KELLY
MAX_KELLY = 0.1  # max 10% of bankroll per bet
df_test["kelly_frac"] = np.minimum(df_test["kelly_frac"], MAX_KELLY)


# -----------------------
# Compute Kelly-scaled P&L
# -----------------------
df_test["pnl_kelly"] = 0.0
df_test.loc[df_test["bet_win"] == 1, "pnl_kelly"] = df_test.loc[df_test["bet_win"] == 1, "kelly_frac"] * PAYOUT
df_test.loc[df_test["bet_win"] == 0, "pnl_kelly"] = -df_test.loc[df_test["bet_win"] == 0, "kelly_frac"]
df_test.loc[df_test["bet_side"] == "NO BET", "pnl_kelly"] = 0

# -----------------------
# Summarize overall Kelly bets
# -----------------------
bets = df_test[df_test["bet_side"] != "NO BET"].copy()
bets["cum_pnl_kelly"] = bets["pnl_kelly"].cumsum()

summary_kelly = {
    "num_bets": len(bets),
    "win_rate": bets["bet_win"].mean(),
    "total_pnl": bets["pnl_kelly"].sum(),
    "roi": bets["pnl_kelly"].sum() / (len(bets)+0.00001),
    "avg_ev": bets["bet_ev"].mean(),
}

print("Kelly Betting Summary:", summary_kelly)


# -----------------------
# Bankroll simulation (compounded Kelly)
# -----------------------
INITIAL_BANKROLL = 1000  # starting bankroll in units
bankroll = INITIAL_BANKROLL
bankroll_history = []

for i, row in bets.iterrows():
    bet_amount = bankroll * row["kelly_frac"]   # fraction of current bankroll
    if row["bet_win"] == 1:
        bankroll += bet_amount * PAYOUT
    else:
        bankroll -= bet_amount
    bankroll_history.append(bankroll)

bets["bankroll"] = bankroll_history

# -----------------------
# Plot compounded bankroll
# -----------------------
plt.figure(figsize=(10,5))
plt.plot(bets["GAME_DATE"], bets["bankroll"])
plt.title("Compounded Bankroll Over Time (Kelly Sizing)")
plt.xlabel("Date")
plt.ylabel("Bankroll")
plt.grid(True)
plt.show()

# -----------------------
# Optional: final bankroll summary
# -----------------------
print(f"Initial bankroll: {INITIAL_BANKROLL}")
print(f"Final bankroll: {bankroll:.2f}")
print(f"Total return: {(bankroll / INITIAL_BANKROLL - 1) * 100:.2f}%")


# -----------------------
# Edge bucket analysis
# -----------------------
bets["edge_bucket"] = pd.cut(
    bets["edge"],
    bins=[-10, -4, -2, 0, 2, 4, 10]
)
edge_summary = bets.groupby("edge_bucket", observed=False).agg(
    num_bets=("pnl_kelly", "count"),
    total_pnl=("pnl_kelly", "sum"),
    roi=("pnl_kelly", lambda x: x.sum() / (x.count()+0.00000001)),
    win_rate=("bet_win", "mean")
).reset_index()
print("\nEdge Bucket Summary (Kelly-Weighted):")
print(edge_summary)

# -----------------------
# Probability bucket analysis
# -----------------------
bets["prob_bucket"] = pd.cut(
    bets["win_prob_home"],
    bins=np.linspace(0, 1, 11)
)
prob_summary = bets.groupby("prob_bucket", observed=False).agg(
    num_bets=("pnl_kelly", "count"),
    total_pnl=("pnl_kelly", "sum"),
    roi=("pnl_kelly", lambda x: x.sum() / (x.count()+0.00000001)),
    win_rate=("bet_win", "mean")
).reset_index()
print("\nProbability Bucket Summary (Kelly-Weighted):")
print(prob_summary)

# -----------------------
# Plot cumulative P&L
# -----------------------
plt.figure(figsize=(10,5))
plt.plot(bets["GAME_DATE"], bets["cum_pnl_kelly"])
plt.title("Cumulative P&L – LightGBM Spread Model (Kelly Sizing)")
plt.xlabel("Date")
plt.ylabel("Units")
plt.grid(True)
plt.show()

# -----------------------
# Save output
# -----------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
bets.to_csv(os.path.join(OUTPUT_DIR, "kelly_bets_summary.csv"), index=False)
print(f"\nKelly betting output saved to {OUTPUT_DIR}/kelly_bets_summary.csv")
