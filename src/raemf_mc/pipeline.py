"""End-to-end RAEMF-MC laptop pipeline."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raemf_mc import CLASS_ORDER, HORIZONS
from raemf_mc.backtest.exposure import backtest_exposure, labels_to_exposure
from raemf_mc.backtest.metrics import backtest_metrics
from raemf_mc.calibration.metrics import multiclass_brier, safe_log_loss
from raemf_mc.calibration.temperature_scaling import apply_temperature, fit_temperature
from raemf_mc.config import write_config_snapshot
from raemf_mc.data.loader import load_price_data, sha256_file
from raemf_mc.evaluation.classification import evaluate_predictions
from raemf_mc.features.selection import select_features
from raemf_mc.features.technical import build_features
from raemf_mc.models.base import fill_features
from raemf_mc.models.ebm_forecaster import EBMForecaster
from raemf_mc.models.macd_baseline import macd_probabilities
from raemf_mc.models.random_forest_forecaster import RandomForestForecaster
from raemf_mc.models.xgboost_forecaster import XGBoostForecaster
from raemf_mc.regime.filtered_hmm import fit_filtered_hmm
from raemf_mc.reporting.plots import (
    plot_backtest,
    plot_class_distribution,
    plot_data_overview,
    plot_hmm_and_risk,
    plot_metric_comparison,
    plot_monte_carlo,
)
from raemf_mc.reporting.report_builder import build_docs_and_readme, build_run_report
from raemf_mc.risk.egarch_t import fit_egarch_features
from raemf_mc.simulation.structural_mc import simulate_paths
from raemf_mc.targets.regime_targets import create_multihorizon_targets
from raemf_mc.uncertainty.block_bootstrap import bootstrap_difference_frame
from raemf_mc.uncertainty.confidence import confidence_label, market_filter
from raemf_mc.uncertainty.drift import feature_drift
from raemf_mc.validation.leakage_checks import (
    assert_no_future_feature_columns,
    assert_target_end_before_boundary,
)
from raemf_mc.validation.purged_split import make_outer_split


def _git_metadata() -> dict[str, object]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "short_sha": run(["git", "rev-parse", "--short", "HEAD"]) or "nogit",
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote": run(["git", "remote", "-v"]),
        "dirty": bool(run(["git", "status", "--short"])),
    }


def _run_dir() -> Path:
    meta = _git_metadata()
    return Path("outputs/runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{meta['short_sha']}"


def _loss_vectors(y: pd.Series, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = pd.Categorical(y.astype(str), categories=CLASS_ORDER).codes
    onehot = np.eye(len(CLASS_ORDER))[codes]
    brier = np.sum((proba - onehot) ** 2, axis=1)
    logloss = -np.log(np.clip(proba[np.arange(len(y)), codes], 1e-9, 1.0))
    return brier, logloss


def _latest_refresh(run_dir: Path) -> None:
    latest = Path("outputs/latest")
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)


def run_pipeline(data_path: str | Path, config: dict[str, Any]) -> Path:
    """Run the full laptop experiment and write artifacts."""
    start = time.time()
    seed = int(config.get("runtime", {}).get("seed", 42))
    np.random.seed(seed)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-raemf")
    run_dir = _run_dir()
    figures = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    df, data_meta = load_price_data(data_path)
    targeted = create_multihorizon_targets(
        df,
        bull_threshold=float(config["target"]["bull_threshold"]),
        bear_threshold=float(config["target"]["bear_threshold"]),
        stress_threshold=float(config["target"]["stress_threshold"]),
        volatility_window=int(config["target"]["volatility_window"]),
    )
    tech, registry = build_features(targeted)
    returns = np.log(targeted["close"] / targeted["close"].shift(1))
    # Fit HMM and EGARCH once on the longest-horizon training region.
    valid60 = targeted[f"target_60"].notna()
    split60 = make_outer_split(
        targeted.loc[valid60, "date"].reset_index(drop=True),
        targeted.loc[valid60, "target_end_date_60"].reset_index(drop=True),
        float(config["split"]["train_fraction"]),
        float(config["split"]["validation_fraction"]),
    )
    train60_global = targeted.loc[valid60].index.to_numpy()[split60.train]
    hmm_res = fit_filtered_hmm(tech, returns, train60_global, int(config["hmm"]["n_states"]), list(config["hmm"]["seeds"]))
    risk_res = fit_egarch_features(returns, train60_global)
    warnings.extend(hmm_res.diagnostics.get("warnings", []))
    warnings.extend(risk_res.diagnostics.get("warnings", []))
    all_features = pd.concat([tech, hmm_res.probabilities, risk_res.features], axis=1)
    assert_no_future_feature_columns(list(all_features.columns))

    registry.to_frame().to_csv(run_dir / "feature_registry.csv", index=False)
    pd.DataFrame({"sha256": [sha256_file(data_path)]}).to_csv(run_dir / "data_sha256.txt", index=False)
    write_config_snapshot(config, run_dir / "config_snapshot.yaml")
    (run_dir / "environment.txt").write_text(
        f"python={platform.python_version()}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )
    (run_dir / "git_metadata.json").write_text(json.dumps(_git_metadata(), indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_rows: list[dict[str, object]] = []
    class_rows: list[pd.DataFrame] = []
    cm_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    hyper: dict[str, Any] = {"hmm": hmm_res.diagnostics, "egarch": risk_res.diagnostics, "calibration": {}}
    bootstrap_rows: list[dict[str, object]] = []
    latest_outlook: dict[str, object] = {
        "as_of_date": targeted["date"].iloc[-1].strftime("%Y-%m-%d"),
        "last_close": float(targeted["close"].iloc[-1]),
        "horizons": {},
        "note": "Không phải lời khuyên đầu tư.",
    }
    model_prob_for_backtest: pd.Series | None = None
    class_distribution_rows: list[dict[str, object]] = []

    for h in HORIZONS:
        valid = targeted[f"target_{h}"].notna()
        sub = targeted.loc[valid].reset_index()
        sub_features = all_features.loc[sub["index"]].reset_index(drop=True)
        split = make_outer_split(
            sub["date"],
            sub[f"target_end_date_{h}"],
            float(config["split"]["train_fraction"]),
            float(config["split"]["validation_fraction"]),
        )
        assert_target_end_before_boundary(sub[f"target_end_date_{h}"], split.train, split.validation_start, f"train h={h}")
        assert_target_end_before_boundary(sub[f"target_end_date_{h}"], split.validation, split.test_start, f"validation h={h}")
        y = sub[f"target_{h}"].astype(str)
        for cls, n in y.value_counts().reindex(CLASS_ORDER).fillna(0).items():
            class_distribution_rows.append({"horizon": h, "class": cls, "count": int(n)})
        selected, removed = select_features(
            sub_features,
            split.train,
            float(config["features"]["missing_threshold"]),
            float(config["features"]["correlation_threshold"]),
        )
        selected_rows.append(pd.DataFrame({"horizon": h, "feature": selected, "status": "selected"}))
        if not removed.empty:
            rr = removed.copy()
            rr.insert(0, "horizon", h)
            rr["status"] = "removed"
            selected_rows.append(rr.rename(columns={"reason": "status_detail"}))
        x_train, x_val, x_test, x_all = fill_features(
            sub_features.loc[split.train, selected],
            sub_features.loc[split.validation, selected],
            sub_features.loc[split.test, selected],
            sub_features[selected],
        )
        y_train = y.iloc[split.train]
        y_val = y.iloc[split.validation]
        y_test = y.iloc[split.test]

        models = {
            "RAEMF-MC": EBMForecaster(seed, **config["models"]["ebm"]).fit(x_train, y_train),
            "XGBoost": XGBoostForecaster(seed, **config["models"]["xgboost"]).fit(x_train, y_train, x_val, y_val),
            "Random Forest": RandomForestForecaster(seed, **config["models"]["random_forest"]).fit(x_train, y_train),
        }
        macd_all = macd_probabilities(sub["close"], sub["target_sigma"]).to_numpy()
        probas: dict[str, dict[str, np.ndarray]] = {"MACD": {"val": macd_all[split.validation], "test": macd_all[split.test], "all": macd_all}}
        for name, model in models.items():
            p_val_raw = model.predict_proba(x_val)
            p_test_raw = model.predict_proba(x_test)
            t, val_loss, use = fit_temperature(p_val_raw, y_val)
            p_val = apply_temperature(p_val_raw, t) if use else p_val_raw
            p_test = apply_temperature(p_test_raw, t) if use else p_test_raw
            p_all_raw = model.predict_proba(x_all)
            p_all = apply_temperature(p_all_raw, t) if use else p_all_raw
            probas[name] = {"val": p_val, "test": p_test, "all": p_all}
            hyper["calibration"][f"{name}_{h}"] = {
                "temperature": t,
                "used": use,
                "validation_log_loss_after": val_loss,
            }
            if isinstance(model, EBMForecaster) and model.warning:
                warnings.append(model.warning)
            if hasattr(model, "importance"):
                model.importance().head(30).to_csv(run_dir / f"feature_importance_{name.replace(' ', '_')}_{h}.csv", index=False)

        for name, parts in probas.items():
            met, cls_met, cm = evaluate_predictions(y_test, parts["test"], name, h)
            metrics_rows.append(met)
            class_rows.append(cls_met)
            cm_rows.append(cm)
            pred = pd.DataFrame(
                {
                    "date": sub["date"].iloc[split.test].to_numpy(),
                    "horizon": h,
                    "model": name,
                    "actual": y_test.to_numpy(),
                    "predicted": [CLASS_ORDER[i] for i in parts["test"].argmax(axis=1)],
                }
            )
            for j, cls in enumerate(CLASS_ORDER):
                pred[f"prob_{cls}"] = parts["test"][:, j]
            prediction_rows.append(pred)

        ra_brier, ra_log = _loss_vectors(y_test, probas["RAEMF-MC"]["test"])
        for bench in ["XGBoost", "Random Forest", "MACD"]:
            b_brier, b_log = _loss_vectors(y_test, probas[bench]["test"])
            bootstrap_rows.append({"horizon": h, "benchmark": bench, "metric": "brier", "diff": ra_brier - b_brier})
            bootstrap_rows.append({"horizon": h, "benchmark": bench, "metric": "log_loss", "diff": ra_log - b_log})

        latest_p = probas["RAEMF-MC"]["all"][-1]
        hmm_ent = float(hmm_res.probabilities["hmm_entropy"].iloc[-1])
        conf = confidence_label(latest_p, hmm_ent)
        mf = market_filter(latest_p, conf, config["market_filter"])
        latest_outlook["horizons"][str(h)] = {
            "probabilities": {cls: float(latest_p[i]) for i, cls in enumerate(CLASS_ORDER)},
            "predicted_class": CLASS_ORDER[int(latest_p.argmax())],
            "confidence": conf,
            "market_filter": mf,
        }
        if h == 20:
            mf_all = [market_filter(row, confidence_label(row, float(hmm_res.probabilities["hmm_entropy"].iloc[sub["index"].iloc[i]])), config["market_filter"]) for i, row in enumerate(probas["RAEMF-MC"]["all"])]
            exposure = labels_to_exposure(pd.Series(mf_all, index=sub["index"]))
            model_prob_for_backtest = exposure

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(run_dir / "metrics_by_model_horizon.csv", index=False)
    pd.concat(class_rows, ignore_index=True).to_csv(run_dir / "class_metrics.csv", index=False)
    pd.concat(cm_rows, ignore_index=True).to_csv(run_dir / "confusion_matrices.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(run_dir / "predictions_test.csv", index=False)
    pd.concat(selected_rows, ignore_index=True).to_csv(run_dir / "selected_features.csv", index=False)
    pd.DataFrame(class_distribution_rows).to_csv(run_dir / "class_distribution.csv", index=False)
    (run_dir / "hyperparameters.json").write_text(json.dumps(hyper, indent=2, ensure_ascii=False), encoding="utf-8")

    boot = bootstrap_difference_frame(
        bootstrap_rows,
        int(config["bootstrap"]["metric_replicates"]),
        int(config["bootstrap"]["block_length"]),
        seed,
    )
    boot.to_csv(run_dir / "bootstrap_differences.csv", index=False)
    boot.rename(columns={"mean_diff": "metric_diff_mean"}).to_csv(run_dir / "metrics_confidence_intervals.csv", index=False)

    transition = np.asarray(hmm_res.diagnostics["transition_matrix"], dtype=float)
    prob_cols = [c for c in hmm_res.probabilities.columns if c.startswith("hmm_prob_state_")]
    state_prob = hmm_res.probabilities[prob_cols].iloc[-1].to_numpy()
    state_mean = np.repeat(float(returns.iloc[train60_global].mean()), len(prob_cols))
    paths_by_h: dict[int, np.ndarray] = {}
    mc_rows = []
    for h in HORIZONS:
        paths, summ = simulate_paths(
            float(targeted["close"].iloc[-1]),
            state_prob,
            transition,
            state_mean,
            float(risk_res.features["egarch_sigma"].iloc[-1]),
            h,
            int(config["monte_carlo"]["paths"]),
            seed,
        )
        paths_by_h[h] = paths
        mc_rows.append(summ)
    pd.concat(mc_rows, ignore_index=True).to_csv(run_dir / "monte_carlo_summary.csv", index=False)

    if model_prob_for_backtest is not None:
        exposure = model_prob_for_backtest.reindex(targeted.index).ffill().fillna(0.25)
        bt = backtest_exposure(targeted["close"], exposure, float(config["backtest"]["transaction_cost_bps"]))
        bt.insert(0, "date", targeted["date"])
        bt.to_csv(run_dir / "backtest_timeseries.csv", index=False)
        pd.DataFrame([backtest_metrics(bt, "RAEMF-MC exposure")]).to_csv(run_dir / "backtest_metrics.csv", index=False)
        plot_backtest(bt, figures)

    drift = feature_drift(all_features.iloc[train60_global].fillna(0), all_features.iloc[-len(split60.test) :].fillna(0))
    drift.to_csv(run_dir / "feature_drift.csv", index=False)
    (run_dir / "latest_outlook.json").write_text(json.dumps(latest_outlook, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "date": latest_outlook["as_of_date"],
                "horizon": h,
                **latest_outlook["horizons"][str(h)]["probabilities"],
                "predicted_class": latest_outlook["horizons"][str(h)]["predicted_class"],
                "confidence": latest_outlook["horizons"][str(h)]["confidence"],
                "market_filter": latest_outlook["horizons"][str(h)]["market_filter"],
            }
            for h in HORIZONS
        ]
    ).to_csv(run_dir / "predictions_latest.csv", index=False)

    plot_data_overview(targeted, figures)
    plot_class_distribution(targeted, figures, HORIZONS)
    plot_metric_comparison(metrics, figures)
    plot_hmm_and_risk(targeted, hmm_res.probabilities, risk_res.features, figures)
    plot_monte_carlo(paths_by_h, figures)
    (run_dir / "warnings.json").write_text(json.dumps(warnings, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "runtime.json").write_text(json.dumps({"seconds": time.time() - start, "rows": len(df), "data_meta": data_meta}, indent=2, ensure_ascii=False), encoding="utf-8")

    build_run_report(run_dir)
    build_docs_and_readme(run_dir)
    _latest_refresh(run_dir)
    return run_dir
