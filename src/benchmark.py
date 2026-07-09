import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from src.data import load_vnindex_csv
from src.features import chronological_split, make_features
from src.metrics import classification_metrics, financial_metrics, regression_metrics
from src.models import (
    feature_importance_rows,
    fit_predict_hmm,
    fit_predict_macd,
    fit_predict_supervised,
    model_specs,
)
from src.plots import (
    plot_equity_curves,
    plot_feature_importance,
    plot_forecast_panel,
    plot_metric_heatmap,
    plot_price_macd,
    setup_plot_style,
)
from src.report import write_readme


def _split_summary(train, valid, test):
    rows = []
    for name, frame in [("train", train), ("valid", valid), ("test", test)]:
        rows.append(
            {
                "split": name,
                "rows": len(frame),
                "start": frame["date"].min().date().isoformat(),
                "end": frame["date"].max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _rank(metrics: pd.DataFrame, financial: pd.DataFrame):
    merged = metrics.merge(financial, on=["horizon", "model"], how="left")
    score_cols = ["balanced_accuracy", "f1", "spearman_ic", "r2", "strategy_sharpe"]
    ranked_parts = []
    for horizon, group in merged.groupby("horizon"):
        scored = group.copy()
        ranks = []
        for col in score_cols:
            values = scored[col].replace([np.inf, -np.inf], np.nan)
            if values.notna().sum() <= 1:
                ranks.append(pd.Series(0.0, index=scored.index))
            else:
                ranks.append(values.rank(pct=True).fillna(0.0))
        scored["rank_score"] = pd.concat(ranks, axis=1).mean(axis=1)
        ranked_parts.append(scored.sort_values("rank_score", ascending=False))
    return pd.concat(ranked_parts, ignore_index=True)


def _make_prediction_rows(test, bundle, horizon):
    signal = pd.Series(bundle.pred_direction, index=test.index).astype(int)
    daily = test["daily_return_next"].fillna(0)
    strategy_return = signal * daily
    pred_return = bundle.pred_return
    if pred_return is None:
        pred_return = np.where(bundle.pred_direction == 1, test[f"future_return_{horizon}d"].mean(), 0.0)
    return pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "horizon": horizon,
            "model": bundle.model,
            "actual_return": test[f"future_return_{horizon}d"].to_numpy(),
            "actual_direction": test[f"future_up_{horizon}d"].astype(int).to_numpy(),
            "pred_return": pred_return,
            "pred_direction": bundle.pred_direction.astype(int),
            "score_up": bundle.score_up if bundle.score_up is not None else np.nan,
            "daily_return_next": daily.to_numpy(),
            "strategy_return": strategy_return.to_numpy(),
            "buy_hold_return": daily.to_numpy(),
        }
    )


def run_benchmark(data_path: Path, output_dir: Path, horizons=(5, 20, 60)):
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)

    raw = load_vnindex_csv(data_path)
    df, feature_cols = make_features(raw, horizons)
    df = df.dropna(subset=feature_cols + [f"future_return_{h}d" for h in horizons] + ["daily_return_next"]).copy()
    train, valid, test = chronological_split(df)
    split_summary = _split_summary(train, valid, test)

    train_valid = pd.concat([train, valid], ignore_index=True)
    specs = model_specs()
    prediction_frames = []
    metric_rows = []
    financial_rows = []
    importance_rows = []
    regime_rows = []

    for horizon in horizons:
        target_return = f"future_return_{horizon}d"
        target_up = f"future_up_{horizon}d"
        x_train = train_valid[feature_cols]
        y_train_ret = train_valid[target_return]
        y_train_up = train_valid[target_up].astype(int)
        x_test = test[feature_cols]
        y_test_ret = test[target_return]
        y_test_up = test[target_up].astype(int)

        bundles = [fit_predict_macd(train_valid, test, horizon)]
        for name, spec in specs.items():
            bundles.append(fit_predict_supervised(name, spec, x_train, y_train_ret, y_train_up, x_test))
        hmm_bundle = fit_predict_hmm(train_valid, test, feature_cols, horizon)
        bundles.append(hmm_bundle)
        for state, mean_return in hmm_bundle.extra["state_mean_return"].items():
            regime_rows.append({"horizon": horizon, "state": state, "mean_forward_return": mean_return})

        for bundle in bundles:
            preds = _make_prediction_rows(test, bundle, horizon)
            prediction_frames.append(preds)
            cls = classification_metrics(y_test_up, bundle.pred_direction, bundle.score_up)
            reg = regression_metrics(y_test_ret, bundle.pred_return)
            metric_rows.append({"horizon": horizon, "model": bundle.model, **cls, **reg})
            fin = financial_metrics(
                preds["date"],
                preds["strategy_return"],
                preds["buy_hold_return"],
                preds["pred_direction"],
            )
            financial_rows.append({"horizon": horizon, "model": bundle.model, **fin})
            importance_rows.extend(feature_importance_rows(bundle, feature_cols, horizon))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    financial = pd.DataFrame(financial_rows)
    feature_importance = pd.DataFrame(importance_rows)
    regime_summary = pd.DataFrame(regime_rows)
    ranking = _rank(metrics, financial)

    raw.to_csv(output_dir / "clean_vnindex_data.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    financial.to_csv(output_dir / "financial_metrics_by_horizon.csv", index=False)
    ranking.to_csv(output_dir / "model_ranking.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)

    setup_plot_style()
    plot_price_macd(df, figures_dir / "01_price_macd_rsi.png")
    plot_metric_heatmap(metrics, "balanced_accuracy", figures_dir / "02_balanced_accuracy_heatmap.png")
    plot_metric_heatmap(financial, "strategy_sharpe", figures_dir / "03_strategy_sharpe_heatmap.png")
    for horizon in horizons:
        plot_equity_curves(predictions, horizon, figures_dir / f"04_equity_curves_{horizon}d.png")
        plot_forecast_panel(predictions, horizon, figures_dir / f"05_forecast_panel_{horizon}d.png")
    plot_feature_importance(feature_importance, figures_dir / "06_feature_importance.png")

    artifacts = {
        "price_macd": "outputs/figures/01_price_macd_rsi.png",
        "balanced_accuracy_heatmap": "outputs/figures/02_balanced_accuracy_heatmap.png",
        "sharpe_heatmap": "outputs/figures/03_strategy_sharpe_heatmap.png",
        "equity_5": "outputs/figures/04_equity_curves_5d.png",
        "equity_20": "outputs/figures/04_equity_curves_20d.png",
        "equity_60": "outputs/figures/04_equity_curves_60d.png",
        "forecast_5": "outputs/figures/05_forecast_panel_5d.png",
        "forecast_20": "outputs/figures/05_forecast_panel_20d.png",
        "forecast_60": "outputs/figures/05_forecast_panel_60d.png",
        "feature_importance": "outputs/figures/06_feature_importance.png",
    }
    write_readme(
        Path("README.md"),
        {
            "rows": len(raw),
            "start": raw["date"].min().date().isoformat(),
            "end": raw["date"].max().date().isoformat(),
        },
        split_summary,
        metrics,
        financial,
        ranking,
        artifacts,
    )

    print("Done. Key artifacts:")
    print(f"- README.md")
    print(f"- {output_dir / 'model_ranking.csv'}")
    print(f"- {output_dir / 'metrics_by_horizon.csv'}")
    print(f"- {output_dir / 'financial_metrics_by_horizon.csv'}")
    print(f"- {figures_dir}")
