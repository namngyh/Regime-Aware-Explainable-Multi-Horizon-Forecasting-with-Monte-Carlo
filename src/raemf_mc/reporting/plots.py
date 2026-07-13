"""Artifact-driven Vietnamese research figures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from raemf_mc import CLASS_ORDER, HORIZONS
from raemf_mc.calibration.metrics import expected_calibration_error, multiclass_brier
from raemf_mc.data.loader import load_price_data
from raemf_mc.targets.regime_targets import create_multihorizon_targets


MODEL_COLORS = {
    "RAEMF-MC": "#0072B2",
    "RAEMF-MC uncalibrated": "#56B4E9",
    "XGBoost (full features)": "#009E73",
    "Random Forest (full features)": "#E69F00",
    "MACD probabilistic": "#CC79A7",
    "MACD deterministic": "#D55E00",
    "Buy-and-Hold": "#000000",
    "Cash": "#777777",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def plot_data_overview(df: pd.DataFrame, figures: Path, split: dict[str, str]) -> None:
    dates = pd.to_datetime(df["date"])
    val_start = pd.Timestamp(split["validation_start"])
    test_start = pd.Timestamp(split["test_start"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axvspan(dates.iloc[0], val_start, color="#56B4E9", alpha=0.10, label="Train")
    ax.axvspan(val_start, test_start, color="#E69F00", alpha=0.12, label="Validation")
    ax.axvspan(test_start, dates.iloc[-1], color="#009E73", alpha=0.10, label="Test")
    ax.plot(dates, df["close"], color="#1f2937", linewidth=1.1, label="VN-Index")
    ax.set(title="VN-Index và phân chia dữ liệu theo thời gian", xlabel="Ngày", ylabel="Điểm số")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, figures / "vnindex_va_phan_chia_du_lieu.png")

    returns = np.log(df["close"] / df["close"].shift(1))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(dates, returns, color="#0072B2", linewidth=0.7)
    ax.axvline(test_start, color="#D55E00", linestyle="--", linewidth=1, label="Bắt đầu test")
    ax.set(title="Lợi suất log VN-Index theo thời gian", xlabel="Ngày", ylabel="Lợi suất log")
    ax.legend(fontsize=8)
    _save(fig, figures / "loi_suat_theo_thoi_gian.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(returns.dropna(), bins=70, color="#56B4E9", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(title="Phân phối lợi suất log VN-Index", xlabel="Lợi suất log", ylabel="Số quan sát")
    _save(fig, figures / "phan_phoi_loi_suat.png")


def plot_class_distribution(targeted: pd.DataFrame, figures: Path) -> None:
    data = pd.DataFrame(
        {
            h: targeted[f"target_{h}"].astype(str).value_counts().reindex(CLASS_ORDER).fillna(0)
            for h in HORIZONS
        }
    ).T
    fig, ax = plt.subplots(figsize=(8, 4.6))
    data.plot(kind="bar", ax=ax, color=["#009E73", "#56B4E9", "#E69F00", "#D55E00"])
    ax.set(title="Phân phối lớp theo chân trời dự báo", xlabel="Chân trời (phiên)", ylabel="Số quan sát")
    ax.legend(title="Lớp", fontsize=8)
    _save(fig, figures / "phan_phoi_lop_theo_horizon.png")


def plot_hmm_and_risk(df: pd.DataFrame, hmm: pd.DataFrame, risk: pd.DataFrame, figures: Path) -> None:
    dates = pd.to_datetime(df["date"])
    probability_columns = [column for column in hmm if column.startswith("hmm_prob_state_")]
    hard = hmm[probability_columns].to_numpy().argmax(axis=1)
    labels = hmm.get("hmm_state_label", pd.Series(probability_columns)).astype(str)
    state_names = []
    for state, column in enumerate(probability_columns):
        state_labels = labels[hard == state]
        state_names.append(str(state_labels.mode().iloc[0]) if not state_labels.empty else column)
    colors = ["#009E73", "#56B4E9", "#E69F00", "#D55E00", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for index, column in enumerate(probability_columns):
        name = state_names[index] if index < len(state_names) else column
        ax.plot(dates, hmm[column], label=name, linewidth=0.8, color=colors[index % len(colors)])
    ax.set(title="Xác suất trạng thái Filtered HMM", xlabel="Ngày", ylabel="Xác suất lọc")
    ax.legend(ncol=min(5, len(probability_columns)), fontsize=8)
    _save(fig, figures / "xac_suat_filtered_hmm.png")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(dates, df["close"], color="#1f2937", linewidth=0.8)
    for state in range(len(probability_columns)):
        mask = hard == state
        ax.scatter(dates[mask], df.loc[mask, "close"], s=5, alpha=0.55, color=colors[state], label=state_names[state])
    ax.set(title="VN-Index và trạng thái HMM đã căn chỉnh", xlabel="Ngày", ylabel="Điểm số")
    ax.legend(ncol=min(4, len(probability_columns)), fontsize=8)
    _save(fig, figures / "hmm_regime_overlay.png")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(dates, risk["egarch_sigma"], color="#D55E00", linewidth=0.9)
    ax.set(title="Biến động điều kiện EGARCH Student-t", xlabel="Ngày", ylabel="Sigma ngày")
    _save(fig, figures / "egarch_conditional_volatility.png")


def plot_metric_comparison(metrics: pd.DataFrame, figures: Path) -> None:
    labels = {
        "macro_f1": "Macro F1 (cao hơn tốt hơn)",
        "balanced_accuracy": "Balanced accuracy (cao hơn tốt hơn)",
        "mcc": "MCC (cao hơn tốt hơn)",
        "brier": "Brier score (thấp hơn tốt hơn)",
        "log_loss": "Log loss (thấp hơn tốt hơn)",
        "ece": "ECE (thấp hơn tốt hơn)",
        "recall_bear": "Recall Bear (cao hơn tốt hơn)",
        "recall_stress": "Recall Stress (cao hơn tốt hơn)",
    }
    for metric, title in labels.items():
        pivot = metrics.pivot_table(index="horizon", columns="model", values=metric)
        colors = [MODEL_COLORS.get(str(model), "#777777") for model in pivot.columns]
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        pivot.plot(kind="bar", ax=ax, color=colors)
        ax.set(title=title, xlabel="Chân trời (phiên)", ylabel=metric)
        ax.legend(title="Mô hình", fontsize=7)
        _save(fig, figures / f"so_sanh_{metric}.png")


def plot_confusion_matrices(confusion: pd.DataFrame, figures: Path) -> None:
    for (model, horizon), frame in confusion.groupby(["model", "horizon"]):
        matrix = frame.set_index("actual")[CLASS_ORDER].to_numpy(dtype=float)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(4):
            for col in range(4):
                ax.text(col, row, f"{int(matrix[row, col])}\n{normalized[row, col]:.0%}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(4), CLASS_ORDER, rotation=30, ha="right")
        ax.set_yticks(range(4), CLASS_ORDER)
        ax.set(title=f"Ma trận nhầm lẫn: {model}, {horizon} phiên", xlabel="Dự báo", ylabel="Thực tế")
        fig.colorbar(image, ax=ax, fraction=0.046)
        _save(fig, figures / f"confusion_matrix_{_slug(str(model))}_{int(horizon)}.png")


def plot_precision_recall(predictions: pd.DataFrame, figures: Path) -> None:
    for horizon, horizon_frame in predictions.groupby("horizon"):
        for target_class in ["Bear", "Stress"]:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            for model, frame in horizon_frame.groupby("model"):
                if model == "RAEMF-MC uncalibrated":
                    continue
                truth = (frame["actual"].astype(str) == target_class).astype(int)
                precision, recall, _ = precision_recall_curve(truth, frame[f"prob_{target_class}"])
                ax.plot(recall, precision, label=model, color=MODEL_COLORS.get(model), linewidth=1.2)
            prevalence = float((horizon_frame.drop_duplicates("date")["actual"] == target_class).mean())
            ax.axhline(prevalence, color="black", linestyle="--", linewidth=0.8, label="Tỷ lệ lớp")
            ax.set(title=f"Precision-recall lớp {target_class}, {int(horizon)} phiên", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
            ax.legend(fontsize=7)
            _save(fig, figures / f"precision_recall_{target_class.lower()}_{int(horizon)}.png")


def _reliability_points(frame: pd.DataFrame, bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = frame[[f"prob_{name}" for name in CLASS_ORDER]].to_numpy()
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    actual = pd.Categorical(frame["actual"], categories=CLASS_ORDER).codes
    correct = (predicted == actual).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ids = np.minimum(np.digitize(confidence, edges[1:-1]), bins - 1)
    mean_conf, accuracy, counts = [], [], []
    for bin_id in range(bins):
        mask = ids == bin_id
        if mask.any():
            mean_conf.append(float(confidence[mask].mean()))
            accuracy.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
    return np.asarray(mean_conf), np.asarray(accuracy), np.asarray(counts)


def plot_reliability(predictions: pd.DataFrame, figures: Path) -> None:
    models = ["RAEMF-MC uncalibrated", "RAEMF-MC", "XGBoost (full features)", "Random Forest (full features)", "MACD probabilistic"]
    for horizon, horizon_frame in predictions.groupby("horizon"):
        fig, (ax, count_ax) = plt.subplots(2, 1, figsize=(8.2, 7), gridspec_kw={"height_ratios": [3, 1]})
        for model in models:
            frame = horizon_frame[horizon_frame["model"] == model]
            if frame.empty:
                continue
            x, y, _ = _reliability_points(frame)
            probability = frame[[f"prob_{name}" for name in CLASS_ORDER]].to_numpy()
            ece = expected_calibration_error(frame["actual"], probability)
            brier = multiclass_brier(frame["actual"], probability)
            ax.plot(x, y, marker="o", markersize=3, label=f"{model}: ECE={ece:.3f}, Brier={brier:.3f}", color=MODEL_COLORS.get(model))
        reference = horizon_frame[horizon_frame["model"] == "RAEMF-MC"]
        x, _, counts = _reliability_points(reference)
        count_ax.bar(x, counts, width=0.07, color="#0072B2", alpha=0.65)
        count_ax.set(xlabel="Xác suất dự báo", ylabel="Số mẫu")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Hiệu chỉnh lý tưởng")
        ax.set(title=f"Reliability diagram, {int(horizon)} phiên", xlabel="Độ tin cậy trung bình", ylabel="Tỷ lệ đúng", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
        _save(fig, figures / f"reliability_diagram_{int(horizon)}.png")


def plot_explainability(run_dir: Path, figures: Path) -> None:
    for horizon in HORIZONS:
        importance_path = run_dir / f"feature_importance_RAEMF-MC_{horizon}.csv"
        if importance_path.exists():
            frame = pd.read_csv(importance_path).dropna(subset=["importance"]).head(20).sort_values("importance")
            fig, ax = plt.subplots(figsize=(8.4, 6.2))
            ax.barh(frame["feature"].astype(str), frame["importance"], color="#0072B2")
            ax.set(title=f"Tầm quan trọng đặc trưng RAEMF-MC, {horizon} phiên", xlabel="Độ quan trọng toàn cục", ylabel="Đặc trưng")
            _save(fig, figures / f"feature_importance_raemf_mc_{horizon}.png")

        shape_path = run_dir / f"ebm_shape_values_{horizon}.csv"
        if shape_path.exists():
            shape = pd.read_csv(shape_path)
            features = shape["feature"].drop_duplicates().head(4).tolist()
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            for ax, feature in zip(axes.flat, features, strict=False):
                part = shape[shape["feature"] == feature]
                for target_class in ["Bull", "Bear", "Stress"]:
                    cls = part[part["class"] == target_class]
                    ax.plot(cls["value"], cls["probability"], label=target_class)
                ax.set(title=str(feature), xlabel="Giá trị đặc trưng", ylabel="Xác suất mô hình")
            for ax in axes.flat[len(features) :]:
                ax.axis("off")
            axes.flat[0].legend(fontsize=7)
            fig.suptitle(f"EBM shape plot từ mô hình, {horizon} phiên", fontsize=12)
            _save(fig, figures / f"ebm_shape_plot_{horizon}.png")

        local_path = run_dir / f"local_explanation_{horizon}.csv"
        if local_path.exists():
            local = pd.read_csv(local_path)
            fig, axes = plt.subplots(1, 3, figsize=(12, 5.6))
            for ax, target_class in zip(axes, ["Bull", "Bear", "Stress"], strict=False):
                part = local[local["class"] == target_class].copy()
                part["absolute"] = part["contribution"].abs()
                part = part.nlargest(10, "absolute").sort_values("contribution")
                ax.barh(part["feature"], part["contribution"], color=np.where(part["contribution"] >= 0, "#009E73", "#D55E00"))
                ax.set(title=target_class, xlabel="Thay đổi xác suất", ylabel="")
            fig.suptitle(f"Đóng góp cục bộ cho dự báo mới nhất, {horizon} phiên", fontsize=12)
            _save(fig, figures / f"local_explanation_latest_{horizon}.png")


def plot_monte_carlo(run_dir: Path, figures: Path) -> None:
    summary = pd.read_csv(run_dir / "monte_carlo_summary.csv").set_index("horizon")
    for horizon in HORIZONS:
        quantiles = pd.read_csv(run_dir / f"monte_carlo_quantiles_{horizon}.csv")
        row = summary.loc[horizon]
        x = quantiles["step"]
        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.fill_between(x, quantiles["q025"], quantiles["q975"], alpha=0.14, color="#56B4E9", label="Khoảng 95%")
        ax.fill_between(x, quantiles["q100"], quantiles["q900"], alpha=0.23, color="#0072B2", label="Khoảng 80%")
        ax.fill_between(x, quantiles["q250"], quantiles["q750"], alpha=0.35, color="#009E73", label="Khoảng 50%")
        ax.plot(x, quantiles["q500"], color="#1f2937", linewidth=1.5, label="Trung vị")
        note = (
            f"P(lợi suất dương)={row['prob_positive']:.1%}; VaR95={row['var_95']:.1%}; "
            f"CVaR95={row['cvar_95']:.1%}\nESS={row['ess']:.0f}; state cuối trội={int(row['dominant_terminal_state'])}"
        )
        ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=8, va="bottom", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"})
        ax.set(title=f"Fan chart Monte Carlo theo chế độ, {horizon} phiên", xlabel="Phiên phía trước", ylabel="Mức VN-Index mô phỏng")
        ax.legend(fontsize=8)
        _save(fig, figures / f"fan_chart_monte_carlo_{horizon}.png")


def plot_backtest(backtest: pd.DataFrame, figures: Path) -> None:
    plots = [
        ("equity", "Đường vốn ngoài mẫu", "Giá trị tương đối", "backtest_equity_oos.png"),
        ("drawdown", "Drawdown ngoài mẫu", "Drawdown", "backtest_drawdown_oos.png"),
        ("exposure", "Exposure thực thi ngoài mẫu", "Exposure", "backtest_exposure_oos.png"),
        ("rolling_sharpe", "Sharpe lăn 63 phiên ngoài mẫu", "Sharpe", "backtest_rolling_sharpe_oos.png"),
        ("cumulative_turnover", "Turnover tích lũy ngoài mẫu", "Turnover", "backtest_turnover_oos.png"),
    ]
    for column, title, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        for strategy, frame in backtest.groupby("strategy"):
            if column in {"exposure", "cumulative_turnover"} and strategy in {"Buy-and-Hold", "Cash"}:
                continue
            ax.plot(pd.to_datetime(frame["date"]), frame[column], label=strategy, color=MODEL_COLORS.get(strategy), linewidth=1.0)
        ax.set(title=title, xlabel="Ngày", ylabel=ylabel)
        ax.legend(fontsize=7, ncol=2)
        _save(fig, figures / filename)


def plot_bootstrap(bootstrap: pd.DataFrame, figures: Path) -> None:
    for metric, frame in bootstrap.groupby("metric"):
        frame = frame.copy().sort_values(["horizon", "benchmark"])
        labels = [f"{int(row.horizon)}p: {row.benchmark}" for row in frame.itertuples()]
        y = np.arange(len(frame))
        fig, ax = plt.subplots(figsize=(9.5, max(4.5, len(frame) * 0.36)))
        ax.errorbar(
            frame["mean_diff"],
            y,
            xerr=[frame["mean_diff"] - frame["ci_low"], frame["ci_high"] - frame["mean_diff"]],
            fmt="o",
            color="#0072B2",
            ecolor="#777777",
            capsize=3,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y, labels)
        ax.set(title=f"Chênh lệch {metric}: RAEMF-MC trừ benchmark", xlabel="Mean difference và khoảng tin cậy bootstrap 95%", ylabel="")
        _save(fig, figures / f"bootstrap_forest_{metric}.png")


def plot_ablation(ablation: pd.DataFrame, figures: Path) -> None:
    pivot = ablation.pivot_table(index="configuration", columns="horizon", values="brier")
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu_r")
    for row in range(len(pivot)):
        for col in range(len(pivot.columns)):
            ax.text(col, row, f"{pivot.iloc[row, col]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.set(title="Ablation theo Brier score (thấp hơn tốt hơn)", xlabel="Chân trời (phiên)", ylabel="Cấu hình")
    fig.colorbar(image, ax=ax, fraction=0.035)
    _save(fig, figures / "ablation_study.png")


def plot_walk_forward(folds: pd.DataFrame, figures: Path) -> None:
    best = folds[folds["is_best_trial"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, ["objective", "brier", "macro_f1", "recall_stress"], strict=False):
        for horizon, frame in best.groupby("horizon"):
            ax.plot(frame["fold"], frame[metric], marker="o", label=f"{int(horizon)} phiên")
        ax.set(title=metric, xlabel="Fold", ylabel=metric)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Độ ổn định theo purged walk-forward fold", fontsize=12)
    _save(fig, figures / "metric_theo_fold_walk_forward.png")


def plot_latest(latest: dict[str, object], figures: Path) -> None:
    rows = []
    for horizon, value in latest["horizons"].items():
        for target_class, probability in value["probabilities"].items():
            rows.append({"horizon": int(horizon), "class": target_class, "probability": probability})
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="horizon", columns="class", values="probability").reindex(columns=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pivot.plot(kind="bar", ax=ax, color=["#009E73", "#56B4E9", "#E69F00", "#D55E00"])
    notes = []
    for horizon in pivot.index:
        item = latest["horizons"][str(horizon)]
        notes.append(f"{horizon}p: {item['predicted_class']}, {item['confidence']}, H={item['entropy']:.2f}, margin={item['margin']:.2f}")
    ax.text(1.02, 0.5, "\n".join(notes), transform=ax.transAxes, va="center", fontsize=8)
    ax.set(title=f"Xác suất dự báo mới nhất tại {latest['as_of_date']}", xlabel="Chân trời (phiên)", ylabel="Xác suất", ylim=(0, 1))
    ax.legend(title="Lớp", fontsize=8)
    _save(fig, figures / "du_bao_moi_nhat_theo_horizon.png")


def generate_all_plots(run_dir: str | Path, data_path: str | Path = "data.csv") -> list[Path]:
    """Regenerate every figure from persisted run artifacts."""
    run_dir = Path(run_dir)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    df, _ = load_price_data(data_path)
    targeted = create_multihorizon_targets(df)
    split = json.loads((run_dir / "split_boundaries.json").read_text(encoding="utf-8"))
    hmm = pd.read_csv(run_dir / "hmm_filtered_probabilities.csv")
    risk = pd.read_csv(run_dir / "egarch_features.csv")
    metrics = pd.read_csv(run_dir / "metrics_by_model_horizon.csv")
    predictions = pd.read_csv(run_dir / "predictions_test.csv")
    confusion = pd.read_csv(run_dir / "confusion_matrices.csv")
    backtest = pd.read_csv(run_dir / "backtest_timeseries.csv")
    bootstrap = pd.read_csv(run_dir / "bootstrap_differences.csv")
    ablation = pd.read_csv(run_dir / "ablation_metrics.csv")
    folds = pd.read_csv(run_dir / "walk_forward_metrics.csv")
    latest = json.loads((run_dir / "latest_outlook.json").read_text(encoding="utf-8"))

    plot_data_overview(targeted, figures, split)
    plot_class_distribution(targeted, figures)
    plot_hmm_and_risk(targeted, hmm, risk, figures)
    plot_metric_comparison(metrics, figures)
    plot_confusion_matrices(confusion, figures)
    plot_precision_recall(predictions, figures)
    plot_reliability(predictions, figures)
    plot_explainability(run_dir, figures)
    plot_monte_carlo(run_dir, figures)
    plot_backtest(backtest, figures)
    plot_bootstrap(bootstrap, figures)
    plot_ablation(ablation, figures)
    plot_walk_forward(folds, figures)
    plot_latest(latest, figures)
    return sorted(figures.glob("*.png"))
