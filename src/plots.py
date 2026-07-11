from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.size": 9,
        }
    )


def plot_price_macd(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(df["date"], df["close"], color="#1f4e79", linewidth=1.2)
    axes[0].set_title("VNIndex close")
    axes[1].plot(df["date"], df["macd"], label="MACD", color="#0b5394")
    axes[1].plot(
        df["date"], df["macd_signal"], label="Signal", color="#cc0000"
    )
    axes[1].bar(
        df["date"],
        df["macd_hist"],
        color=np.where(df["macd_hist"] >= 0, "#6aa84f", "#e06666"),
        alpha=0.65,
    )
    axes[1].legend(loc="upper left")
    axes[2].plot(df["date"], df["rsi_14"], color="#674ea7", label="RSI 14")
    axes[2].axhline(70, color="#cc0000", linestyle="--", linewidth=0.8)
    axes[2].axhline(30, color="#38761d", linestyle="--", linewidth=0.8)
    axes[2].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_heatmap(metrics: pd.DataFrame, metric: str, output_path: Path):
    pivot = metrics.pivot_table(index="model", columns="horizon", values=metric)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cmap = "RdYlGn" if metric == "forecast_score" else "RdYlGn_r"
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap=cmap, ax=ax)
    ax.set_title(f"Out-of-sample {metric} - horizon 20")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_forecast_panel(predictions: pd.DataFrame, horizon: int, output_path: Path):
    models = list(predictions["model"].drop_duplicates())
    fig, axes = plt.subplots(len(models), 1, figsize=(13, 7), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        group = predictions[predictions["model"] == model].sort_values("date")
        ax.plot(
            group["date"],
            group["actual_return"],
            color="#555555",
            linewidth=0.9,
            label="Actual 20-session return",
        )
        ax.plot(
            group["date"],
            group["pred_return"],
            color="#0b5394",
            linewidth=1.0,
            label="Predicted return",
        )
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.set_title(model)
        ax.legend(loc="upper left")
    fig.suptitle(f"Locked-test forecasts - horizon {horizon} sessions")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_future_return_forecast(future: pd.DataFrame, output_path: Path):
    data = future.copy()
    data["pred_return_pct"] = data["pred_return"] * 100
    colors = np.where(data["pred_return"] >= 0, "#38761d", "#cc0000")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(data["model"], data["pred_return_pct"], color=colors)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f%%")
    ax.set_ylabel("Predicted 20-session return (%)")
    ax.set_title(f"Forecast as of {data['as_of_date'].iloc[0]}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_future_price_targets(future: pd.DataFrame, output_path: Path):
    data = future.copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(data["model"], data["predicted_close"], color="#3c78d8")
    latest_close = data["latest_close"].iloc[0]
    ax.axhline(
        latest_close,
        color="#222222",
        linestyle="--",
        label=f"Latest close {latest_close:,.2f}",
    )
    ax.bar_label(bars, fmt="%.2f")
    ax.set_ylabel("Projected VNIndex close")
    ax.set_title(f"20-session target around {data['target_date'].iloc[0]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path):
    data = feature_importance.sort_values("importance", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["feature"], data["importance"], color="#3c78d8")
    ax.set_xlabel("Random Forest importance")
    ax.set_title("Top 15 features for the 20-session return forecast")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_bootstrap_intervals(bootstrap: pd.DataFrame, output_path: Path):
    data = bootstrap[bootstrap["metric"] == "forecast_score"].copy()
    data = data.sort_values("estimate")
    lower = data["estimate"] - data["ci_lower_95"]
    upper = data["ci_upper_95"] - data["estimate"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(
        data["estimate"],
        data["model"],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#0b5394",
        ecolor="#6d9eeb",
        capsize=5,
    )
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Forecast score with 95% moving-block bootstrap CI")
    ax.set_title("Uncertainty of locked-test model quality")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_future_projection_path(
    price_history: pd.DataFrame,
    future: pd.DataFrame,
    output_path: Path,
):
    history = price_history.dropna(subset=["date", "close"]).tail(252)
    latest_date = pd.to_datetime(future["as_of_date"].iloc[0])
    latest_close = future["latest_close"].iloc[0]
    target_date = pd.to_datetime(future["target_date"].iloc[0])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history["date"], history["close"], color="#1f4e79", label="VNIndex close")
    palette = sns.color_palette("Set2", n_colors=len(future))
    for color, (_, row) in zip(palette, future.iterrows()):
        ax.plot(
            [latest_date, target_date],
            [latest_close, row["predicted_close"]],
            marker="o",
            linewidth=2,
            color=color,
            label=f"{row['model']}: {row['predicted_close']:,.2f}",
        )
    ax.axvline(latest_date, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_title("VNIndex history and 20-session model projections")
    ax.set_ylabel("Index level")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_actual_vs_predicted(predictions: pd.DataFrame, output_path: Path):
    models = list(predictions["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    limits = [
        min(predictions["actual_return"].min(), predictions["pred_return"].min()),
        max(predictions["actual_return"].max(), predictions["pred_return"].max()),
    ]
    for ax, model in zip(axes, models):
        data = predictions[predictions["model"] == model]
        ax.scatter(
            data["actual_return"],
            data["pred_return"],
            s=12,
            alpha=0.35,
            color="#0b5394",
        )
        ax.plot(limits, limits, color="#cc0000", linestyle="--", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("Actual 20-session return")
    axes[0].set_ylabel("Predicted 20-session return")
    fig.suptitle("Actual versus predicted return on the locked test")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_residual_diagnostics(predictions: pd.DataFrame, output_path: Path):
    data = predictions.copy()
    data["residual"] = data["actual_return"] - data["pred_return"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model, group in data.groupby("model"):
        sns.kdeplot(group["residual"], label=model, ax=axes[0], fill=False)
    axes[0].axvline(0, color="#777777", linestyle="--", linewidth=0.9)
    axes[0].set_title("Residual distributions")
    axes[0].legend()
    sns.boxplot(data=data, x="model", y="residual", ax=axes[1])
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set_title("Residual spread and outliers")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_yearly_score_heatmap(stability: pd.DataFrame, output_path: Path):
    pivot = stability.pivot(index="model", columns="year", values="forecast_score")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Forecast-score stability across locked-test years")
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_quantile_future_band(
    price_history: pd.DataFrame,
    future: pd.DataFrame,
    output_path: Path,
):
    quantile = future.dropna(
        subset=["predicted_close_q10", "predicted_close_q90"]
    ).iloc[0]
    history = price_history.dropna(subset=["date", "close"]).tail(126)
    latest_date = pd.to_datetime(quantile["as_of_date"])
    target_date = pd.to_datetime(quantile["target_date"])
    latest_close = quantile["latest_close"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(history["date"], history["close"], color="#1f4e79", label="VNIndex close")
    ax.plot(
        [latest_date, target_date],
        [latest_close, quantile["predicted_close"]],
        color="#38761d",
        marker="o",
        linewidth=2,
        label="Hybrid q50",
    )
    ax.fill_between(
        [latest_date, target_date],
        [latest_close, quantile["predicted_close_q10"]],
        [latest_close, quantile["predicted_close_q90"]],
        color="#93c47d",
        alpha=0.3,
        label="Hybrid q10-q90",
    )
    ax.set_title("HMM-EGARCH-LightGBM quantile projection")
    ax.set_ylabel("Index level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_quantile_test_intervals(predictions: pd.DataFrame, output_path: Path):
    data = predictions.dropna(
        subset=["pred_return_q10", "pred_return_q90"]
    ).sort_values("date").tail(220)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(
        data["date"],
        data["pred_return_q10"],
        data["pred_return_q90"],
        color="#93c47d",
        alpha=0.3,
        label="q10-q90 interval",
    )
    ax.plot(data["date"], data["pred_return"], color="#38761d", label="q50")
    ax.plot(
        data["date"],
        data["actual_return"],
        color="#444444",
        linewidth=0.9,
        label="Actual return",
    )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_title("Hybrid quantile calibration on the recent locked test")
    ax.set_ylabel("20-session return")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ensemble_weight_curve(trials: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(trials["hmm_weight"], trials["cv_robust_score"], color="#0b5394")
    selected = trials[trials["selected"]].iloc[0]
    ax.scatter(
        [selected["hmm_weight"]],
        [selected["cv_robust_score"]],
        color="#cc0000",
        s=70,
        label=(
            f"Selected HMM={selected['hmm_weight']:.0%}, "
            f"RF={selected['random_forest_weight']:.0%}"
        ),
    )
    ax.set_xlabel("HMM weight")
    ax.set_ylabel("OOF robust forecast score")
    ax.set_title("Out-of-fold HMM-Random Forest ensemble weights")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_hybrid_feature_importance(feature_importance: pd.DataFrame, output_path: Path):
    data = feature_importance.sort_values("importance", ascending=True).tail(18)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(data["feature"], data["importance"], color="#6aa84f")
    ax.set_xlabel("LightGBM split importance")
    ax.set_title("Hybrid HMM-EGARCH-LightGBM feature importance")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
