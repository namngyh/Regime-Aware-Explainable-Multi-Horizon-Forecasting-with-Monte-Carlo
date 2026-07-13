"""Matplotlib figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raemf_mc import CLASS_ORDER


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_data_overview(df: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["date"], df["close"], label="VN-Index")
    ax.set_title("VN-Index theo thời gian")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Điểm số")
    ax.legend()
    _save(fig, figures / "vnindex_theo_thoi_gian.png")

    ret = np.log(df["close"] / df["close"].shift(1))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(df["date"], ret, label="Lợi suất log")
    ax.set_title("Lợi suất log theo thời gian")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Lợi suất")
    ax.legend()
    _save(fig, figures / "loi_suat_theo_thoi_gian.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ret.dropna(), bins=60)
    ax.set_title("Phân phối lợi suất log")
    ax.set_xlabel("Lợi suất")
    ax.set_ylabel("Tần suất")
    _save(fig, figures / "phan_phoi_loi_suat.png")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(df["date"], ret.rolling(40).std(), label="Biến động lăn 40 phiên")
    ax.set_title("Biến động lăn")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Độ lệch chuẩn")
    ax.legend()
    _save(fig, figures / "bien_dong_lan.png")


def plot_class_distribution(targeted: pd.DataFrame, figures: Path, horizons: list[int]) -> None:
    rows = []
    for h in horizons:
        vc = targeted[f"target_{h}"].astype(str).value_counts().reindex(CLASS_ORDER).fillna(0)
        for cls, n in vc.items():
            rows.append({"horizon": h, "class": cls, "n": n})
    dd = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in CLASS_ORDER:
        s = dd[dd["class"] == cls]
        ax.bar([str(x) for x in s["horizon"]], s["n"], label=cls, alpha=0.75)
    ax.set_title("Phân phối lớp theo chân trời")
    ax.set_xlabel("Chân trời")
    ax.set_ylabel("Số quan sát")
    ax.legend()
    _save(fig, figures / "phan_phoi_lop_theo_horizon.png")


def plot_metric_comparison(metrics: pd.DataFrame, figures: Path) -> None:
    for metric in ["macro_f1", "balanced_accuracy", "brier", "log_loss", "recall_bear", "recall_stress"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot = metrics.pivot_table(index="horizon", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"So sánh {metric}")
        ax.set_xlabel("Chân trời")
        ax.set_ylabel(metric)
        ax.legend(title="Mô hình", fontsize=8)
        _save(fig, figures / f"so_sanh_{metric}.png")


def plot_hmm_and_risk(df: pd.DataFrame, hmm: pd.DataFrame, risk: pd.DataFrame, figures: Path) -> None:
    cols = [c for c in hmm.columns if c.startswith("hmm_prob_state_")]
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in cols:
        ax.plot(df["date"], hmm[c], label=c)
    ax.set_title("Xác suất filtered state theo thời gian")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Xác suất")
    ax.legend(fontsize=7)
    _save(fig, figures / "xac_suat_filtered_hmm.png")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(df["date"], risk["egarch_sigma"], label="EGARCH sigma")
    ax.set_title("Biến động điều kiện EGARCH Student-t")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Sigma")
    ax.legend()
    _save(fig, figures / "egarch_conditional_volatility.png")


def plot_monte_carlo(paths_by_horizon: dict[int, np.ndarray], figures: Path) -> None:
    for h, paths in paths_by_horizon.items():
        q = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        x = np.arange(paths.shape[1])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(x, q[0], q[4], alpha=0.20, label="5-95%")
        ax.fill_between(x, q[1], q[3], alpha=0.35, label="25-75%")
        ax.plot(x, q[2], label="Trung vị")
        ax.set_title(f"Fan chart Monte Carlo {h} phiên")
        ax.set_xlabel("Phiên phía trước")
        ax.set_ylabel("Mức chỉ số mô phỏng")
        ax.legend()
        _save(fig, figures / f"fan_chart_monte_carlo_{h}.png")


def plot_backtest(bt: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    equity = np.exp(bt["strategy_return"].fillna(0).cumsum())
    ax.plot(bt["date"], equity, label="RAEMF-MC exposure")
    ax.set_title("Đường tăng trưởng tài sản minh họa")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá trị tương đối")
    ax.legend()
    _save(fig, figures / "backtest_duong_tang_truong.png")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.step(bt["date"], bt["exposure"], where="post", label="Exposure")
    ax.set_title("Exposure theo thời gian")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Exposure")
    ax.legend()
    _save(fig, figures / "backtest_exposure.png")
