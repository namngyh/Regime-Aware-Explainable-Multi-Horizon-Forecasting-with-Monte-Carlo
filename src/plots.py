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
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    axes[0].plot(df["date"], df["close"], color="#1f4e79", linewidth=1.2, label="VNIndex Close")
    bullish = df["macd_bullish"] == 1
    axes[0].fill_between(df["date"], df["close"].min(), df["close"].max(), where=bullish, color="#d9ead3", alpha=0.35)
    axes[0].set_title("VNIndex va vung MACD bullish")
    axes[0].legend(loc="upper left")
    axes[1].plot(df["date"], df["macd"], label="MACD", color="#0b5394")
    axes[1].plot(df["date"], df["macd_signal"], label="Signal", color="#cc0000")
    axes[1].bar(df["date"], df["macd_hist"], color=np.where(df["macd_hist"] >= 0, "#6aa84f", "#e06666"), alpha=0.7)
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
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=pivot.stack().median(), ax=ax)
    ax.set_title(f"So sanh {metric} theo horizon")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_equity_curves(predictions: pd.DataFrame, horizon: int, output_path: Path):
    subset = predictions[predictions["horizon"] == horizon].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    for model, group in subset.groupby("model"):
        daily = group.sort_values("date")["strategy_return"].fillna(0)
        equity = (1 + daily).cumprod()
        ax.plot(group.sort_values("date")["date"], equity, label=model, linewidth=1.1)
    bh = subset[subset["model"] == subset["model"].iloc[0]].sort_values("date")
    ax.plot(bh["date"], (1 + bh["buy_hold_return"].fillna(0)).cumprod(), label="Buy & Hold", color="black", linewidth=1.6)
    ax.set_title(f"Equity curve tren tap test - horizon {horizon} phien")
    ax.set_ylabel("Growth of 1")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_forecast_panel(predictions: pd.DataFrame, horizon: int, output_path: Path):
    subset = predictions[predictions["horizon"] == horizon].copy()
    models = list(subset["model"].drop_duplicates())
    ncols = 2
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, nrows * 3)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, model in zip(axes, models):
        group = subset[subset["model"] == model].sort_values("date")
        ax.plot(group["date"], group["actual_return"], color="#444444", linewidth=0.9, label="Actual future return")
        ax.plot(group["date"], group["pred_return"], color="#0b5394", linewidth=0.9, label="Predicted return")
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.fill_between(
            group["date"],
            group["actual_return"].min(),
            group["actual_return"].max(),
            where=group["pred_direction"].astype(bool),
            color="#d9ead3",
            alpha=0.35,
        )
        ax.set_title(model)
    for ax in axes[len(models) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"Du bao return va tin hieu long/flat - horizon {horizon} phien", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_path)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path):
    if feature_importance.empty:
        return
    top = (
        feature_importance.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(18)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, y="feature", x="importance", color="#3c78d8", ax=ax)
    ax.set_title("Top feature importance trung binh qua cac mo hinh cay/boosting")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
