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
