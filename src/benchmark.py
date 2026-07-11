import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from src.data import load_vnindex_csv
from src.ensemble import combine_hmm_random_forest, tune_oof_hmm_rf_ensemble
from src.features import chronological_split, make_features
from src.hybrid import fit_predict_hybrid_quantile
from src.hybrid_tuning import tune_hybrid_quantile
from src.metrics import classification_metrics, financial_metrics, regression_metrics
from src.models import (
    fit_predict_hmm,
    fit_predict_random_forest,
    fit_predict_svr,
)
from src.plots import (
    plot_actual_vs_predicted,
    plot_bootstrap_intervals,
    plot_forecast_panel,
    plot_feature_importance,
    plot_future_price_targets,
    plot_future_projection_path,
    plot_future_return_forecast,
    plot_hybrid_feature_importance,
    plot_metric_heatmap,
    plot_ensemble_weight_curve,
    plot_price_macd,
    plot_residual_diagnostics,
    plot_quantile_future_band,
    plot_quantile_test_intervals,
    plot_yearly_score_heatmap,
    setup_plot_style,
)
from src.report import write_readme
from src.research import block_bootstrap_intervals, pairwise_dm_tests
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
            "pred_return_q10": bundle.extra.get("q10", np.full(len(test), np.nan)),
            "pred_return_q90": bundle.extra.get("q90", np.full(len(test), np.nan)),
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


def _quantile_metrics(predictions):
    data = predictions.dropna(subset=["pred_return_q10", "pred_return_q90"])
    if data.empty:
        return pd.DataFrame()

    def pinball(actual, forecast, alpha):
        error = actual - forecast
        return np.mean(np.maximum(alpha * error, (alpha - 1) * error))

    rows = []
    for model, group in data.groupby("model"):
        actual = group["actual_return"].to_numpy()
        q10 = group["pred_return_q10"].to_numpy()
        q50 = group["pred_return"].to_numpy()
        q90 = group["pred_return_q90"].to_numpy()
        rows.append(
            {
                "model": model,
                "coverage_80": np.mean((actual >= q10) & (actual <= q90)),
                "mean_interval_width": np.mean(q90 - q10),
                "pinball_q10": pinball(actual, q10, 0.1),
                "pinball_q50": pinball(actual, q50, 0.5),
                "pinball_q90": pinball(actual, q90, 0.9),
            }
        )
    return pd.DataFrame(rows)


def _future_forecasts(
    full_df,
    feature_cols,
    horizon,
    hmm_params,
    svr_params,
    random_forest_params,
    egarch_params,
    lightgbm_params,
    ensemble_weights,
    ranking,
):
    target = f"future_return_{horizon}d"
    train = full_df.dropna(subset=feature_cols + [target]).copy()
    latest = full_df.dropna(subset=feature_cols).iloc[[-1]].copy()
    hmm_future_bundle = fit_predict_hmm(
        train,
        latest,
        feature_cols,
        horizon,
        hmm_params=hmm_params,
    )
    svr_future_bundle = fit_predict_svr(
        train,
        latest,
        feature_cols,
        horizon,
        params=svr_params,
    )
    forest_future_bundle = fit_predict_random_forest(
        train,
        latest,
        feature_cols,
        horizon,
        params=random_forest_params,
    )
    hybrid_future_bundle = fit_predict_hybrid_quantile(
        train,
        latest,
        feature_cols,
        horizon,
        hmm_params,
        egarch_params,
        lightgbm_params,
    )
    ensemble_future_bundle = combine_hmm_random_forest(
        hmm_future_bundle, forest_future_bundle, ensemble_weights
    )
    bundles = [
        hmm_future_bundle,
        svr_future_bundle,
        forest_future_bundle,
        hybrid_future_bundle,
        ensemble_future_bundle,
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
                "pred_return_q10": (
                    float(bundle.extra["q10"][0])
                    if "q10" in bundle.extra
                    else np.nan
                ),
                "pred_return_q90": (
                    float(bundle.extra["q90"][0])
                    if "q90" in bundle.extra
                    else np.nan
                ),
                "predicted_close_q10": (
                    latest_close * (1 + float(bundle.extra["q10"][0]))
                    if "q10" in bundle.extra
                    else np.nan
                ),
                "predicted_close_q90": (
                    latest_close * (1 + float(bundle.extra["q90"][0]))
                    if "q90" in bundle.extra
                    else np.nan
                ),
                "test_forecast_score": quality["forecast_score"],
                "test_mae": quality["mae"],
                "test_rmse": quality["rmse"],
                "test_balanced_accuracy": quality["balanced_accuracy"],
                "current_regime": (
                    int(bundle.extra["predicted_states"][0])
                    if bundle.model == "HMM Regime"
                    else np.nan
                ),
            }
        )
    current_state = int(hmm_future_bundle.extra["predicted_states"][0])
    regime_summary = pd.DataFrame(
        [
            {
                "state": state,
                "full_train_observations": hmm_future_bundle.extra[
                    "state_counts"
                ].get(state, 0),
                "mean_forward_return": mean_return,
                "is_current_regime": state == current_state,
            }
            for state, mean_return in sorted(
                hmm_future_bundle.extra["state_mean_return"].items()
            )
        ]
    )
    return pd.DataFrame(rows), regime_summary


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

    print("Optimizing HMM, SVR and Random Forest for the 20-session horizon...")
    (
        hmm_params,
        svr_params,
        random_forest_params,
        tuning_trials,
        best_parameters,
    ) = tune_horizon(train_valid, feature_cols, horizon)

    print("Optimizing HMM-EGARCH-LightGBM Quantile...")
    (
        egarch_params,
        lightgbm_params,
        hybrid_tuning_trials,
        hybrid_best,
    ) = tune_hybrid_quantile(
        train_valid,
        feature_cols,
        horizon,
        hmm_params,
    )

    print("Learning OOF HMM-Random Forest ensemble weights...")
    (
        ensemble_weights,
        ensemble_weight_trials,
        ensemble_oof_predictions,
    ) = tune_oof_hmm_rf_ensemble(
        train_valid,
        feature_cols,
        horizon,
        hmm_params,
        random_forest_params,
    )

    hmm_bundle = fit_predict_hmm(
        train_valid,
        test,
        feature_cols,
        horizon,
        hmm_params=hmm_params,
    )
    svr_bundle = fit_predict_svr(
        train_valid,
        test,
        feature_cols,
        horizon,
        params=svr_params,
    )
    random_forest_bundle = fit_predict_random_forest(
        train_valid,
        test,
        feature_cols,
        horizon,
        params=random_forest_params,
    )
    hybrid_bundle = fit_predict_hybrid_quantile(
        train_valid,
        test,
        feature_cols,
        horizon,
        hmm_params,
        egarch_params,
        lightgbm_params,
    )
    ensemble_bundle = combine_hmm_random_forest(
        hmm_bundle,
        random_forest_bundle,
        ensemble_weights,
    )
    bundles = [
        hmm_bundle,
        svr_bundle,
        random_forest_bundle,
        hybrid_bundle,
        ensemble_bundle,
    ]
    predictions = pd.concat(
        [_prediction_frame(test, bundle, horizon) for bundle in bundles],
        ignore_index=True,
    )
    metrics, financial, ranking = _evaluate(predictions)
    stability = _yearly_stability(predictions)
    bootstrap = block_bootstrap_intervals(
        predictions, n_bootstrap=1000, block_size=horizon
    )
    dm_tests = pairwise_dm_tests(predictions, horizon=horizon)
    quantile_metrics = _quantile_metrics(predictions)
    feature_importance = random_forest_bundle.extra["feature_importance"].head(20).copy()
    feature_importance.insert(0, "model", "Random Forest")
    feature_importance.insert(0, "horizon", horizon)
    hybrid_feature_importance = hybrid_bundle.extra["feature_importance"].head(25).copy()
    hybrid_feature_importance.insert(
        0, "model", "HMM-EGARCH-LightGBM Quantile"
    )
    hybrid_feature_importance.insert(0, "horizon", horizon)
    future, hmm_regime_summary = _future_forecasts(
        full_df,
        feature_cols,
        horizon,
        hmm_params,
        svr_params,
        random_forest_params,
        egarch_params,
        lightgbm_params,
        ensemble_weights,
        ranking,
    )

    raw.to_csv(output_dir / "clean_vnindex_data.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    financial.to_csv(output_dir / "financial_metrics.csv", index=False)
    ranking.to_csv(output_dir / "model_ranking.csv", index=False)
    stability.to_csv(output_dir / "test_stability_by_year.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)
    dm_tests.to_csv(output_dir / "pairwise_dm_tests.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    hybrid_feature_importance.to_csv(
        output_dir / "hybrid_feature_importance.csv", index=False
    )
    quantile_metrics.to_csv(output_dir / "quantile_metrics.csv", index=False)
    hmm_regime_summary.to_csv(output_dir / "hmm_regime_summary.csv", index=False)
    tuning_trials.to_csv(output_dir / "tuning_trials.csv", index=False)
    best_parameters.to_csv(output_dir / "best_hyperparameters.csv", index=False)
    hybrid_tuning_trials.to_csv(
        output_dir / "hybrid_tuning_trials.csv", index=False
    )
    pd.DataFrame([hybrid_best]).to_csv(
        output_dir / "hybrid_best_hyperparameters.csv", index=False
    )
    ensemble_weight_trials.to_csv(
        output_dir / "ensemble_weight_trials.csv", index=False
    )
    ensemble_oof_predictions.to_csv(
        output_dir / "ensemble_oof_predictions.csv", index=False
    )
    pd.DataFrame([ensemble_weights]).to_csv(
        output_dir / "ensemble_best_weights.csv", index=False
    )
    future.to_csv(output_dir / "future_forecasts.csv", index=False)

    setup_plot_style()
    plot_price_macd(full_df.dropna(subset=feature_cols), figures_dir / "01_vnindex_context.png")
    plot_metric_heatmap(metrics, "forecast_score", figures_dir / "02_model_score.png")
    plot_metric_heatmap(metrics, "mae", figures_dir / "03_return_mae.png")
    plot_forecast_panel(predictions, horizon, figures_dir / "04_test_forecasts_20d.png")
    plot_future_return_forecast(future, figures_dir / "05_future_return_20d.png")
    plot_future_price_targets(future, figures_dir / "06_future_price_20d.png")
    plot_feature_importance(
        feature_importance, figures_dir / "07_random_forest_feature_importance.png"
    )
    plot_bootstrap_intervals(
        bootstrap, figures_dir / "08_bootstrap_forecast_score.png"
    )
    plot_future_projection_path(
        raw, future, figures_dir / "09_future_projection_path.png"
    )
    plot_actual_vs_predicted(
        predictions, figures_dir / "10_actual_vs_predicted.png"
    )
    plot_residual_diagnostics(
        predictions, figures_dir / "11_residual_diagnostics.png"
    )
    plot_yearly_score_heatmap(
        stability, figures_dir / "12_yearly_score_heatmap.png"
    )
    plot_quantile_future_band(
        raw, future, figures_dir / "13_hybrid_future_quantile_band.png"
    )
    plot_quantile_test_intervals(
        predictions, figures_dir / "14_hybrid_test_quantiles.png"
    )
    plot_ensemble_weight_curve(
        ensemble_weight_trials, figures_dir / "15_oof_ensemble_weights.png"
    )
    plot_hybrid_feature_importance(
        hybrid_feature_importance,
        figures_dir / "16_hybrid_feature_importance.png",
    )

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
        bootstrap=bootstrap,
        dm_tests=dm_tests,
        feature_importance=feature_importance,
        hybrid_feature_importance=hybrid_feature_importance,
        quantile_metrics=quantile_metrics,
        hybrid_tuning_trials=hybrid_tuning_trials,
        hybrid_best=pd.DataFrame([hybrid_best]),
        ensemble_weight_trials=ensemble_weight_trials,
        ensemble_weights=pd.DataFrame([ensemble_weights]),
        hmm_regime_summary=hmm_regime_summary,
        best_parameters=best_parameters,
        tuning_trials=tuning_trials,
        future=future,
    )
    print("Done. Compared 5 models including hybrid quantile and OOF ensemble.")
