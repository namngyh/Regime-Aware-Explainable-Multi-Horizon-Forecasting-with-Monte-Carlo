import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from src.data import load_vnindex_csv
from src.features import chronological_split, make_features
from src.metrics import classification_metrics, financial_metrics, regression_metrics
from src.models import (
    fit_predict_hmm,
    fit_predict_macd,
    macd_bullish_signal,
)
from src.plots import (
    plot_forecast_panel,
    plot_future_price_targets,
    plot_future_return_forecast,
    plot_metric_heatmap,
    plot_price_macd,
    setup_plot_style,
)
from src.report import write_readme
from src.tuning import forecast_score, tune_horizon


def _split_summary(train, valid, test):
    return pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(frame),
                "start": frame["date"].min().date().isoformat(),
                "end": frame["date"].max().date().isoformat(),
            }
            for name, frame in [("train", train), ("valid", valid), ("test", test)]
        ]
    )


def _add_selected_macd(frame, params):
    result = frame.copy()
    result["macd_selected"] = macd_bullish_signal(
        result["close"], params["fast"], params["slow"], params["signal"]
    ).to_numpy()
    return result


def _prediction_frame(test, bundle, horizon):
    actual_return = test[f"future_return_{horizon}d"].to_numpy()
    current_close = test["close"].to_numpy()
    daily_return = test["daily_return_next"].fillna(0).to_numpy()
    return pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "horizon": horizon,
            "model": bundle.model,
            "actual_return": actual_return,
            "pred_return": bundle.pred_return,
            "actual_direction": test[f"future_up_{horizon}d"].astype(int).to_numpy(),
            "pred_direction": bundle.pred_direction,
            "score_up": bundle.score_up,
            "current_close": current_close,
            "actual_close": current_close * (1 + actual_return),
            "predicted_close": current_close * (1 + bundle.pred_return),
            "daily_return_next": daily_return,
            "strategy_return": bundle.pred_direction * daily_return,
            "buy_hold_return": daily_return,
        }
    )


def _evaluate(predictions):
    metrics_rows = []
    financial_rows = []
    for (horizon, model), group in predictions.groupby(["horizon", "model"]):
        classification = classification_metrics(
            group["actual_direction"], group["pred_direction"], group["score_up"]
        )
        regression = regression_metrics(
            group["actual_return"], group["pred_return"], group["current_close"]
        )
        score = forecast_score(classification, regression)
        metrics_rows.append(
            {
                "horizon": horizon,
                "model": model,
                "forecast_score": score,
                **classification,
                **regression,
            }
        )
        financial_rows.append(
            {
                "horizon": horizon,
                "model": model,
                **financial_metrics(
                    group["date"],
                    group["strategy_return"],
                    group["buy_hold_return"],
                    group["pred_direction"],
                ),
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    financial = pd.DataFrame(financial_rows)
    ranking = metrics.merge(financial, on=["horizon", "model"])
    ranking["rank"] = ranking.groupby("horizon")["forecast_score"].rank(
        method="first", ascending=False
    ).astype(int)
    ranking = ranking.sort_values(["horizon", "rank"]).reset_index(drop=True)
    return metrics, financial, ranking


def _yearly_stability(predictions):
    frame = predictions.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows = []
    for (model, year), group in frame.groupby(["model", "year"]):
        classification = classification_metrics(
            group["actual_direction"], group["pred_direction"], group["score_up"]
        )
        regression = regression_metrics(
            group["actual_return"], group["pred_return"], group["current_close"]
        )
        rows.append(
            {
                "model": model,
                "year": year,
                "rows": len(group),
                "forecast_score": forecast_score(classification, regression),
                "mae": regression["mae"],
                "rmse": regression["rmse"],
                "price_mae": regression["price_mae"],
                "price_rmse": regression["price_rmse"],
                "balanced_accuracy": classification["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def _future_forecasts(
    full_df,
    feature_cols,
    horizon,
    macd_params,
    hmm_params,
    ranking,
):
    target = f"future_return_{horizon}d"
    train = full_df.dropna(subset=feature_cols + [target]).copy()
    latest = full_df.dropna(subset=feature_cols).iloc[[-1]].copy()
    macd_frame = _add_selected_macd(full_df, macd_params)
    macd_train = train.copy()
    macd_latest = latest.copy()
    macd_train["macd_selected"] = macd_frame.loc[macd_train.index, "macd_selected"]
    macd_latest["macd_selected"] = macd_frame.loc[
        macd_latest.index, "macd_selected"
    ]
    bundles = [
        fit_predict_macd(macd_train, macd_latest, horizon),
        fit_predict_hmm(
            train,
            latest,
            feature_cols,
            horizon,
            hmm_params=hmm_params,
        ),
    ]
    latest_date = latest["date"].iloc[0]
    latest_close = latest["close"].iloc[0]
    rows = []
    for bundle in bundles:
        quality = ranking.loc[ranking["model"] == bundle.model].iloc[0]
        pred_return = float(bundle.pred_return[0])
        rows.append(
            {
                "as_of_date": latest_date.date().isoformat(),
                "target_date": (
                    latest_date + pd.offsets.BDay(horizon)
                ).date().isoformat(),
                "horizon": horizon,
                "model": bundle.model,
                "latest_close": latest_close,
                "pred_return": pred_return,
                "pred_direction": int(bundle.pred_direction[0]),
                "predicted_close": latest_close * (1 + pred_return),
                "test_forecast_score": quality["forecast_score"],
                "test_mae": quality["mae"],
                "test_rmse": quality["rmse"],
                "test_balanced_accuracy": quality["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def run_benchmark(data_path: Path, output_dir: Path, horizons=(20,)):
    if tuple(horizons) != (20,):
        raise ValueError("This cleaned pipeline is intentionally fixed at horizon 20.")
    horizon = 20
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)

    raw = load_vnindex_csv(data_path)
    full_df, feature_cols = make_features(raw, horizons)
    historical = full_df.dropna(
        subset=feature_cols + [f"future_return_{horizon}d", "daily_return_next"]
    ).copy()
    train, valid, test = chronological_split(historical)
    train_valid = pd.concat([train, valid], ignore_index=True)
    split_summary = _split_summary(train, valid, test)

    print("Optimizing MACD and HMM for the 20-session horizon...")
    macd_params, hmm_params, tuning_trials, best_parameters = tune_horizon(
        train_valid, feature_cols, horizon
    )

    macd_all = _add_selected_macd(historical, macd_params).set_index("date")
    macd_train = train_valid.copy()
    macd_test = test.copy()
    macd_train["macd_selected"] = macd_train["date"].map(
        macd_all["macd_selected"]
    )
    macd_test["macd_selected"] = macd_test["date"].map(
        macd_all["macd_selected"]
    )
    bundles = [
        fit_predict_macd(macd_train, macd_test, horizon),
        fit_predict_hmm(
            train_valid,
            test,
            feature_cols,
            horizon,
            hmm_params=hmm_params,
        ),
    ]
    predictions = pd.concat(
        [_prediction_frame(test, bundle, horizon) for bundle in bundles],
        ignore_index=True,
    )
    metrics, financial, ranking = _evaluate(predictions)
    stability = _yearly_stability(predictions)
    future = _future_forecasts(
        full_df,
        feature_cols,
        horizon,
        macd_params,
        hmm_params,
        ranking,
    )

    raw.to_csv(output_dir / "clean_vnindex_data.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    financial.to_csv(output_dir / "financial_metrics.csv", index=False)
    ranking.to_csv(output_dir / "model_ranking.csv", index=False)
    stability.to_csv(output_dir / "test_stability_by_year.csv", index=False)
    tuning_trials.to_csv(output_dir / "tuning_trials.csv", index=False)
    best_parameters.to_csv(output_dir / "best_hyperparameters.csv", index=False)
    future.to_csv(output_dir / "future_forecasts.csv", index=False)

    setup_plot_style()
    plot_price_macd(full_df.dropna(subset=feature_cols), figures_dir / "01_vnindex_context.png")
    plot_metric_heatmap(metrics, "forecast_score", figures_dir / "02_model_score.png")
    plot_metric_heatmap(metrics, "mae", figures_dir / "03_return_mae.png")
    plot_forecast_panel(predictions, horizon, figures_dir / "04_test_forecasts_20d.png")
    plot_future_return_forecast(future, figures_dir / "05_future_return_20d.png")
    plot_future_price_targets(future, figures_dir / "06_future_price_20d.png")

    write_readme(
        Path("README.md"),
        data_summary={
            "rows": len(raw),
            "start": raw["date"].min().date().isoformat(),
            "end": raw["date"].max().date().isoformat(),
        },
        split_summary=split_summary,
        ranking=ranking,
        stability=stability,
        best_parameters=best_parameters,
        tuning_trials=tuning_trials,
        future=future,
    )
    print("Done. Retained models: MACD 8-24-9 and HMM Regime.")
