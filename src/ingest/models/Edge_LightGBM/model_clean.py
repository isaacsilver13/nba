import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

class SpreadBacktester:
    """
    A backtester for betting on NBA spreads using a machine learning model.

    This class allows loading data, training LightGBM models on past seasons,
    predicting spreads, computing expected value (EV), applying Kelly sizing, 
    simulating bets, and analyzing walk-forward performance.
    """

    def __init__(
        self,
        path: str,
        features: list,
        target: str = "home_margin",
        payout: float = 0.909,
        ev_threshold: float = 0.02,
        fractional_kelly: float = 0.5,
        max_kelly: float = 0.1,
        test_season: int = 2025,
        lgb_params: dict = None,
        initial_bankroll: float = 10000
    ):
        """
        Initialize the backtester with model parameters, features, and betting parameters.

        Parameters
        ----------
        path : str
            Path to the CSV dataset containing game and spread data.
        features : list
            List of column names to use as features in the model.
        target : str
            Column name of the target variable (default: "home_margin").
        payout : float
            Payout multiplier for winning bets (default: 0.909 for -110 odds).
        ev_threshold : float
            Minimum expected value required to place a bet.
        fractional_kelly : float
            Fraction of full Kelly to use for bet sizing (0 < fractional_kelly <= 1).
        max_kelly : float
            Maximum Kelly fraction allowed per bet.
        test_season : int
            Season to hold out for testing.
        lgb_params : dict
            Parameters for LightGBM model.
        initial_bankroll : float
            Starting bankroll for simulation.
        """
        self.features = sorted(features)
        self.target = target
        self.payout = payout
        self.ev_threshold = ev_threshold
        self.fractional_kelly = fractional_kelly
        self.max_kelly = max_kelly
        self.test_season = test_season

        self.lgb_params = lgb_params or {}
        self.model = None
        self.sigma_hat = None
        self.path = path

        self.df = None
        self.df_test = None
        self.bets = None
        self.initial_bankroll = initial_bankroll
        self.fold_results = []
        self.oof_predictions = []
        self.equity_curve = []

    # -----------------------
    # Data handling
    # -----------------------
    def load_data(self, path: str) -> pd.DataFrame:
        """
        Load the CSV dataset and convert relevant columns to numeric types.

        Parameters
        ----------
        path : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Sorted dataframe with numeric features and target.
        """
        df = pd.read_csv(path, parse_dates=["date_x"])
        df = df.rename(columns={"date_x": "GAME_DATE"})

        # Ensure all features and target are numeric
        for col in self.features + [self.target]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        self.df = df.sort_values("GAME_DATE").reset_index(drop=True)
        return self.df

    def train_test_split(self):
        """
        Split data into training (all seasons before test_season) and test set.

        Returns
        -------
        tuple
            X_train, y_train, X_test, y_test dataframes/series
        """
        train = self.df[self.df["season"] < self.test_season]
        test = self.df[self.df["season"] == self.test_season]

        X_train = train[self.features]
        y_train = train[self.target]
        X_test = test[self.features]
        y_test = test[self.target]

        self.df_test = test.copy()
        return X_train, y_train, X_test, y_test

    # -----------------------
    # Model
    # -----------------------
    def fit_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit a LightGBM regressor to training data and compute residual std deviation.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.

        Returns
        -------
        tuple
            Trained LightGBM model, estimated residual std deviation
        """
        self.model = lgb.LGBMRegressor(**self.lgb_params)
        self.model.fit(X_train, y_train)

        residuals = y_train - self.model.predict(X_train)
        self.sigma_hat = residuals.std()

        return self.model, self.sigma_hat

    def predict(self, test_df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Predict target values for a test set and store predictions in df_test.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test data to predict on.
        features : list
            Features to use for prediction.

        Returns
        -------
        pd.DataFrame
            Test dataframe with a "pred" column for predicted margins.
        """
        test_df = test_df.copy()
        test_df["pred"] = self.model.predict(test_df[features])
        self.df_test = test_df  # assign for downstream methods
        return test_df

    # -----------------------
    # Probabilities & EV
    # -----------------------
    def compute_probabilities(self) -> pd.DataFrame:
        """
        Compute standardized edge and implied win probability for home team.

        Returns
        -------
        pd.DataFrame
            df_test updated with 'edge', 'edge_std', 'win_prob_home'
        """
        self.df_test["edge"] = self.df_test["pred"] - self.df_test["spread_signed"]
        self.df_test["edge_std"] = self.df_test["edge"] / self.sigma_hat
        self.df_test["win_prob_home"] = norm.cdf(self.df_test["edge_std"])
        return self.df_test

    def compute_ev(self) -> pd.DataFrame:
        """
        Compute expected value (EV) for both home and away bets.

        Returns
        -------
        pd.DataFrame
            df_test updated with 'ev_home' and 'ev_away'
        """
        p = self.df_test["win_prob_home"]
        self.df_test["ev_home"] = p * self.payout - (1 - p)
        self.df_test["ev_away"] = (1 - p) * self.payout - p
        return self.df_test

    def choose_bets(self) -> pd.DataFrame:
        """
        Decide which side to bet on based on EV threshold.

        Returns
        -------
        pd.DataFrame
            df_test updated with 'bet_side' column ("HOME", "AWAY", "NO BET")
        """
        self.df_test["bet_side"] = np.where(
            self.df_test["ev_home"] > self.ev_threshold,
            "HOME",
            np.where(
                self.df_test["ev_away"] > self.ev_threshold,
                "AWAY",
                "NO BET",
            ),
        )
        return self.df_test

    def walk_forward_splits(
        self,
        time_col: str,
        train_size: int,
        test_size: int,
        df: pd.DataFrame = None
    ):
        """
        Generator for walk-forward train/test splits.

        Parameters
        ----------
        time_col : str
            Column to sort by (usually date).
        train_size : int
            Number of rows in each training fold.
        test_size : int
            Number of rows in each testing fold.
        df : pd.DataFrame, optional
            Dataframe to split (default: self.df)

        Yields
        ------
        tuple
            (train_df, test_df) for each fold
        """
        df = df.copy() if df is not None else self.df.copy()
        df = df.sort_values(time_col).reset_index(drop=True)

        start = 0
        while True:
            train_end = start + train_size
            test_end = train_end + test_size
            if test_end > len(df):
                break
            yield df.iloc[start:train_end], df.iloc[train_end:test_end]
            start += test_size

    # -----------------------
    # Walk-forward simulation
    # -----------------------
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        features: list,
        target: str,
        time_col: str,
        train_size: int,
        test_size: int,
    ) -> pd.DataFrame:
        """
        Perform walk-forward training and betting simulation.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset
        features : list
            Features for model training
        target : str
            Target variable
        time_col : str
            Column for chronological sorting
        train_size : int
            Training window size
        test_size : int
            Test window size

        Returns
        -------
        pd.DataFrame
            Summary metrics per fold
        """
        self.fold_results = []
        self.oof_predictions = []
        all_bets = []
        bankroll = self.initial_bankroll

        df = df.sort_values(time_col).reset_index(drop=True)
        self.df = df.copy()

        for fold, (train_df, test_df) in enumerate(
            self.walk_forward_splits(time_col, train_size, test_size, df=df),
            start=1
        ):
            print(f"Fold {fold}: Train {len(train_df)} rows, Test {len(test_df)} rows")
            # Train model
            self.fit_model(train_df[features], train_df[target])
            # Predict
            test_df = self.predict(test_df, features)
            self.oof_predictions.append(test_df[[time_col, "pred", target]])
            # Betting
            self.compute_probabilities()
            self.compute_ev()
            self.choose_bets()
            self.apply_kelly()
            self.resolve_bets()
            # Fold metrics
            fold_summary = self.fold_metrics(self.df_test, fold, bankroll)
            bankroll = fold_summary["ending_bankroll"]
            self.fold_results.append(fold_summary)
            self.equity_curve.append(bankroll)

        print("Walk forward complete")
        return pd.DataFrame(self.fold_results)

    # -----------------------
    # Betting calculations
    # -----------------------
    @staticmethod
    def _kelly_fraction(p: float, b: float) -> float:
        """
        Compute full Kelly fraction for a single bet.

        Parameters
        ----------
        p : float
            Probability of winning
        b : float
            Payout odds

        Returns
        -------
        float
            Kelly fraction (clipped to [0,1])
        """
        f = (p * (b + 1) - 1) / b
        return np.clip(f, 0, 1)

    def apply_kelly(self) -> pd.DataFrame:
        """
        Apply fractional Kelly sizing to df_test bets.

        Returns
        -------
        pd.DataFrame
            df_test updated with 'kelly_frac'
        """
        df = self.df_test
        df["kelly_frac"] = 0.0

        home = df["bet_side"] == "HOME"
        away = df["bet_side"] == "AWAY"

        df.loc[home, "kelly_frac"] = self._kelly_fraction(
            df.loc[home, "win_prob_home"], self.payout
        )
        df.loc[away, "kelly_frac"] = self._kelly_fraction(
            1 - df.loc[away, "win_prob_home"], self.payout
        )

        df["kelly_frac"] *= self.fractional_kelly
        df["kelly_frac"] = np.minimum(df["kelly_frac"], self.max_kelly)

        self.df_test = df
        return df

    def resolve_bets(self) -> pd.DataFrame:
        """
        Determine which bets won, compute P&L for Kelly-sized bets, and cumulative P&L.

        Returns
        -------
        pd.DataFrame
            Filtered df_test with bets placed, P&L columns, and cumulative P&L
        """
        df = self.df_test

        df["home_margin_needed"] = np.where(
            df["is_home_favorite"] == 1,
            df["spread"],
            -df["spread"],
        )

        df["did_home_cover"] = df["home_margin"] > df["home_margin_needed"]
        df["team_that_covered"] = np.where(df["did_home_cover"], "HOME", "AWAY")
        df["bet_win"] = (df["bet_side"] == df["team_that_covered"]).astype(int)

        df["pnl_kelly"] = 0.0
        df.loc[df["bet_win"] == 1, "pnl_kelly"] = df["kelly_frac"] * self.payout
        df.loc[df["bet_win"] == 0, "pnl_kelly"] = -df["kelly_frac"]

        self.bets = df[df["bet_side"] != "NO BET"].copy()
        self.bets["cum_pnl"] = self.bets["pnl_kelly"].cumsum()
        return self.bets

    # -----------------------
    # Reporting & metrics
    # -----------------------
    def summary(self) -> dict:
        """
        Compute summary statistics for all placed bets.

        Returns
        -------
        dict
            num_bets, win_rate, total_pnl, roi, avg_ev
        """
        b = self.bets
        return {
            "num_bets": len(b),
            "win_rate": b["bet_win"].mean(),
            "total_pnl": b["pnl_kelly"].sum(),
            "roi": b["pnl_kelly"].sum() / max(len(b), 1),
            "avg_ev": b["ev_home"].mean(),
        }

    def plot_pnl(self):
        """Plot cumulative P&L of bets over time."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.bets["GAME_DATE"], self.bets["cum_pnl"])
        plt.title("Cumulative P&L (Kelly Sizing)")
        plt.xlabel("Date")
        plt.ylabel("Units")
        plt.grid(True)
        plt.show()

    def fold_metrics(self, df: pd.DataFrame, fold: int, starting_bankroll: float) -> dict:
        """
        Compute P&L, ROI, and bankroll for a fold.

        Parameters
        ----------
        df : pd.DataFrame
            Test fold dataframe.
        fold : int
            Fold number.
        starting_bankroll : float
            Bankroll at start of fold.

        Returns
        -------
        dict
            Metrics for fold: bets, pnl, roi, starting_bankroll, ending_bankroll
        """
        pnl = df["pnl_kelly"].sum()
        ending_bankroll = starting_bankroll + pnl
        roi = pnl / starting_bankroll if starting_bankroll > 0 else 0
        df["bet"] = (df["bet_side"] != "NO BET").astype("int64")

        return {
            "fold": fold,
            "bets": df["bet"].sum(),
            "pnl": pnl,
            "roi": roi,
            "starting_bankroll": starting_bankroll,
            "ending_bankroll": ending_bankroll,
        }

    def compute_drawdown(self) -> dict:
        """
        Compute max drawdown and drawdown series from equity curve.

        Returns
        -------
        dict
            max_drawdown and drawdown_series
        """
        equity = pd.Series(self.equity_curve)
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return {
            "max_drawdown": drawdown.min(),
            "drawdown_series": drawdown,
        }

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Compute Sharpe ratio of fold ROI.

        Parameters
        ----------
        risk_free_rate : float
            Risk-free rate for excess returns.

        Returns
        -------
        float
            Sharpe ratio
        """
        df = pd.DataFrame(self.fold_results)
        returns = df["roi"]
        excess = returns - risk_free_rate
        return 0.0 if returns.std() == 0 else excess.mean() / excess.std()

    def volatility(self) -> float:
        """
        Compute standard deviation of fold ROI.

        Returns
        -------
        float
        """
        df = pd.DataFrame(self.fold_results)
        return df["roi"].std()

    def oof_calibration(self, n_bins: int = 10) -> pd.DataFrame:
        """
        Perform out-of-fold calibration analysis.

        Parameters
        ----------
        n_bins : int
            Number of bins to split predicted values.

        Returns
        -------
        pd.DataFrame
            Mean predicted vs mean actual per bin
        """
        oof = pd.concat(self.oof_predictions)
        target = self.target
        oof["bin"] = pd.qcut(oof["pred"], n_bins, duplicates="drop")
        calib = (
            oof.groupby("bin", observed=False)
            .agg(
                mean_pred=("pred", "mean"),
                mean_actual=(target, "mean"),
                count=("pred", "size"),
            )
            .reset_index()
        )
        return calib

    def walk_forward_param_sweep(
        self,
        df: pd.DataFrame,
        param_grid: list,
        **walk_forward_kwargs
    ) -> pd.DataFrame:
        """
        Sweep over multiple LightGBM parameter configurations with walk-forward testing.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to use.
        param_grid : list
            List of parameter dictionaries to test.
        walk_forward_kwargs : dict
            Arguments for run_walk_forward

        Returns
        -------
        pd.DataFrame
            Summary of results per parameter configuration
        """
        results = []

        for params in param_grid:
            self.lgb_params |= params
            self.run_walk_forward(df, **walk_forward_kwargs)
            summary = {
                "params": params,
                "final_bankroll": self.equity_curve[-1],
                "max_drawdown": self.compute_drawdown()["max_drawdown"],
                "sharpe": self.sharpe_ratio(),
            }
            results.append(summary)

        return pd.DataFrame(results)




LGB_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.02,
    "max_depth": 4,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1
}


FEATURES = [

"spread_signed", "is_home_favorite", "home_margin_last5",
"away_margin_last5", "home_b2b", "away_b2b","pts_diff_last5",
"margin_diff_last5", "rest_diff", "elo_diff", "home_elo_roll_2Y",
"away_elo_roll_2Y", "rolling_margin_diff_5", "rolling_margin_diff_10",
"rest_advantage", "rest_advantage_sq", "top8_points_diff", "travel_diff_3d",
"travel_diff_7d", "travel_diff_10d", "travel_1d_z", "travel_3d_z", "travel_7d_z",
"travel_10d_z", "fatigue_diff",

"home_team_net_points_r5", "home_team_net_rebounds_r5", "home_team_net_assists_r5",
"home_team_net_STL_r5", "home_team_net_BLK_r5", "home_team_net_turnovers_r5",
"home_team_net_OREB_r5", "home_team_net_fgm_r5", "home_team_net_fga_r5",
"home_team_net_three_pm_r5", "home_team_net_three_pa_r5", "home_team_net_ftm_r5",
"home_team_net_fta_r5",

"home_team_net_points_r10", "home_team_net_rebounds_r10", "home_team_net_assists_r10",
"home_team_net_STL_r10", "home_team_net_BLK_r10", "home_team_net_turnovers_r10",
"home_team_net_OREB_r10", "home_team_net_fgm_r10", "home_team_net_fga_r10",
"home_team_net_three_pm_r10", "home_team_net_three_pa_r10", "home_team_net_ftm_r10",
"home_team_net_fta_r10",

"away_team_net_points_r5", "away_team_net_rebounds_r5", "away_team_net_assists_r5",
"away_team_net_STL_r5", "away_team_net_BLK_r5", "away_team_net_turnovers_r5",
"away_team_net_OREB_r5", "away_team_net_fgm_r5", "away_team_net_fga_r5", 
"away_team_net_three_pm_r5", "away_team_net_three_pa_r5", "away_team_net_ftm_r5",
"away_team_net_fta_r5",

"away_team_net_points_r10", "away_team_net_rebounds_r10", "away_team_net_assists_r10",
"away_team_net_STL_r10", "away_team_net_BLK_r10", "away_team_net_turnovers_r10",
"away_team_net_OREB_r10", "away_team_net_fgm_r10", "away_team_net_fga_r10",
"away_team_net_three_pm_r10", "away_team_net_three_pa_r10", "away_team_net_ftm_r10",
"away_team_net_fta_r10",
]
# optional: keep deterministic ordering
FEATURES = sorted(FEATURES)

TARGET = "home_margin"
TIME_COL = "GAME_DATE"

backtester = SpreadBacktester(
    features=FEATURES,
    path = "data/processed/df_model_3.csv",
    payout=0.909,
    ev_threshold=0.05,
    fractional_kelly=1.5,
    max_kelly=0.5,
    test_season=2025,
    lgb_params=LGB_PARAMS,
    initial_bankroll = 10000
)

# Load data
df = backtester.load_data(backtester.path)

# Run walk-forward
fold_results_df = backtester.run_walk_forward(
    df=df,
    features=FEATURES,
    target=TARGET,
    time_col=TIME_COL,
    train_size=2000,
    test_size=300,
)

# Inspect fold metrics
print(fold_results_df)

# Summary stats
# print(backtester.summary())

# Final bankroll / P&L
# print(backtester.equity_curve[-1])

# Plot cumulative P&L
backtester.plot_pnl()

# Risk metrics
# print("Max drawdown:", backtester.compute_drawdown()["max_drawdown"])
# print("Sharpe ratio:", backtester.sharpe_ratio())
# print("Volatility:", backtester.volatility())
backtester.df_test.to_csv("outputs/model clean output.csv", index=False)
