from pathlib import Path

import pandas as pd


def _table(frame, columns, digits=4):
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: f"{value:.{digits}f}" if pd.notna(value) else "NA"
            )
    view = view.astype(str)
    widths = {
        column: max(len(column), *(len(value) for value in view[column].tolist()))
        for column in view.columns
    }
    header = "| " + " | ".join(
        column.ljust(widths[column]) for column in view.columns
    ) + " |"
    separator = "| " + " | ".join(
        "-" * widths[column] for column in view.columns
    ) + " |"
    rows = [
        "| "
        + " | ".join(row[column].ljust(widths[column]) for column in view.columns)
        + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_readme(
    output_path: Path,
    data_summary: dict,
    split_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    stability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    dm_tests: pd.DataFrame,
    feature_importance: pd.DataFrame,
    hybrid_feature_importance: pd.DataFrame,
    quantile_metrics: pd.DataFrame,
    hybrid_tuning_trials: pd.DataFrame,
    hybrid_best: pd.DataFrame,
    ensemble_weight_trials: pd.DataFrame,
    ensemble_weights: pd.DataFrame,
    hmm_regime_summary: pd.DataFrame,
    best_parameters: pd.DataFrame,
    tuning_trials: pd.DataFrame,
    future: pd.DataFrame,
):
    ranking_view = ranking[
        [
            "rank",
            "model",
            "forecast_score",
            "mae",
            "rmse",
            "r2",
            "spearman_ic",
            "price_mae",
            "price_rmse",
            "balanced_accuracy",
            "strategy_sharpe",
        ]
    ]
    parameter_view = best_parameters[
        [
            "model",
            "candidate_id",
            "cv_score",
            "cv_score_std",
            "cv_robust_score",
            "selected_params",
        ]
    ]
    hybrid_best_view = hybrid_best[
        [
            "egarch_params",
            "lightgbm_params",
            "cv_score",
            "cv_score_std",
            "cv_robust_score",
            "cv_mae",
            "cv_rmse",
        ]
    ]
    ensemble_view = ensemble_weights[
        [
            "hmm_weight",
            "random_forest_weight",
            "cv_score",
            "cv_score_std",
            "cv_robust_score",
            "cv_rmse",
        ]
    ]
    bootstrap_view = bootstrap[bootstrap["metric"] == "forecast_score"][
        ["model", "estimate", "ci_lower_95", "ci_upper_95", "block_size"]
    ]
    dm_view = dm_tests[
        [
            "model_a",
            "model_b",
            "dm_stat",
            "holm_p_value",
            "lower_loss_model",
            "significant_5pct",
        ]
    ]
    rf_importance_view = feature_importance.head(12)[["feature", "importance"]]
    hybrid_importance_view = hybrid_feature_importance.head(15)[
        ["feature", "importance"]
    ]
    regime_view = hmm_regime_summary.copy()
    regime_view["mean_forward_return"] = regime_view["mean_forward_return"].map(
        lambda value: f"{value:.2%}"
    )
    future_view = future[
        [
            "as_of_date",
            "target_date",
            "model",
            "pred_return",
            "predicted_close",
            "predicted_close_q10",
            "predicted_close_q90",
            "pred_direction",
            "test_forecast_score",
        ]
    ].copy()
    future_view["pred_return"] = future_view["pred_return"].map(
        lambda value: f"{value:.2%}"
    )

    latest_close = future["latest_close"].iloc[0]
    winner = ranking.iloc[0]["model"]
    best_mae = ranking.loc[ranking["mae"].idxmin(), "model"]
    best_rmse = ranking.loc[ranking["rmse"].idxmin(), "model"]
    best_price = ranking.loc[ranking["price_mae"].idxmin(), "model"]
    best_direction = ranking.loc[ranking["balanced_accuracy"].idxmax(), "model"]
    significant_pairs = int(dm_tests["significant_5pct"].sum())
    current_regime = hmm_regime_summary.loc[
        hmm_regime_summary["is_current_regime"]
    ].iloc[0]
    current_regime_share = (
        current_regime["full_train_observations"]
        / hmm_regime_summary["full_train_observations"].sum()
    )
    base_seconds = tuning_trials["fit_seconds"].sum()
    hybrid_fit_seconds = hybrid_tuning_trials["fit_seconds"].sum()
    hybrid_prep_seconds = (
        hybrid_tuning_trials.groupby("egarch_candidate_id")[
            "feature_preparation_seconds"
        ].first().sum()
    )

    def stance(value):
        if value >= 0.03:
            return "bullish mạnh"
        if value >= 0.005:
            return "bullish nhẹ"
        if value > -0.005:
            return "trung tính"
        return "bearish"

    stance_summary = "; ".join(
        f"{row.model}: {stance(row.pred_return)} ({row.pred_return:+.2%})"
        for row in future.itertuples()
    )
    forecast_spread = future["pred_return"].max() - future["pred_return"].min()

    body = f"""# VNIndex 20-session Forecast: Hybrid Quantile and OOF Ensemble

Nghiên cứu so sánh năm mô hình: **HMM Regime**, **SVR**, **Random Forest**, **HMM–EGARCH–LightGBM Quantile** và **OOF HMM–Random Forest Ensemble**. Tập test được khóa khỏi toàn bộ quá trình tuning và học trọng số ensemble.

## Kết luận chính

- Mô hình đứng đầu theo `forecast_score`: **{winner}**.
- Return MAE thấp nhất: **{best_mae}**; RMSE thấp nhất: **{best_rmse}**.
- Price MAE thấp nhất: **{best_price}**; Balanced Accuracy cao nhất: **{best_direction}**.
- Có **{significant_pairs}/{len(dm_tests)}** so sánh cặp đạt ý nghĩa 5% sau Holm.

## Kết quả test khóa

{_table(ranking_view, list(ranking_view.columns))}

`forecast_score` gồm 65% dự báo lợi suất, 25% dự báo giá và 10% dự báo hướng. Skill score so với dự báo không đổi (`return=0`, future price bằng current price).

### Cách đọc kết quả

- Hybrid chỉ được xem là cải thiện nếu point score, bootstrap CI và DM test đều ủng hộ; một metric tốt đơn lẻ không đủ.
- OOF ensemble học trọng số duy nhất từ expanding-fold predictions. Test không tham gia chọn trọng số.
- R²/IC âm cho thấy timing và xếp hạng biên độ còn yếu ngay cả khi MAE cải thiện.

## Tối ưu ba mô hình nền

{_table(parameter_view, list(parameter_view.columns))}

- {len(tuning_trials)} cấu hình, 6-fold expanding `TimeSeriesSplit`, `gap=20`.
- Robust selection: `CV mean - 0.25 × CV standard deviation`.
- Thời gian fit cộng dồn: {base_seconds:.1f} giây.

## HMM–EGARCH–LightGBM Quantile

Hybrid dùng causal HMM state probabilities, EGARCH conditional volatility và technical features. LightGBM dự báo ba quantile `q10/q50/q90`; `q50` là point forecast, còn `q10-q90` là vùng bất định 80% dự kiến.

{_table(hybrid_best_view, list(hybrid_best_view.columns))}

- {len(hybrid_tuning_trials)} tổ hợp EGARCH–LightGBM.
- Feature preparation: {hybrid_prep_seconds:.1f} giây; LightGBM fits: {hybrid_fit_seconds:.1f} giây.

### Quantile calibration

{_table(quantile_metrics, list(quantile_metrics.columns))}

Coverage gần 80% là mong muốn. Interval quá rộng có coverage cao nhưng ít hữu ích; cần đọc coverage cùng `mean_interval_width` và pinball loss.

### Hybrid feature importance

{_table(hybrid_importance_view, ["feature", "importance"])}

## OOF HMM–Random Forest Ensemble

{_table(ensemble_view, list(ensemble_view.columns))}

Trọng số được chọn trên lưới 0–100% HMM theo robust OOF score. Nếu trọng số dồn về một biên, dữ liệu OOF không chứng minh mô hình còn lại bổ sung đủ thông tin.

## Chia dữ liệu

{_table(split_summary, ["split", "rows", "start", "end"])}

Dữ liệu có {data_summary['rows']:,} quan sát từ {data_summary['start']} đến {data_summary['end']}. Target là return sau 20 phiên.

## Bootstrap uncertainty

{_table(bootstrap_view, list(bootstrap_view.columns))}

## Diebold–Mariano/Holm pairwise tests

{_table(dm_view, list(dm_view.columns))}

`significant_5pct=True` mới cho phép kết luận squared forecast error khác nhau sau hiệu chỉnh nhiều kiểm định.

## Độ ổn định theo năm

{_table(stability, ["model", "year", "rows", "forecast_score", "mae", "rmse", "price_mae", "balanced_accuracy"])}

## HMM regimes

{_table(regime_view, ["state", "full_train_observations", "mean_forward_return", "is_current_regime"])}

Current regime chiếm {current_regime_share:.1%} lịch sử có nhãn. State labels được lấy từ chính causal HMM refit dùng cho forecast tương lai.

## Random Forest feature importance

{_table(rf_importance_view, ["feature", "importance"])}

## Dự báo VNIndex tương lai

Phiên gần nhất là `{future['as_of_date'].iloc[0]}`, VNIndex đóng cửa **{latest_close:,.2f}**. Target khoảng `{future['target_date'].iloc[0]}`.

{_table(future_view, list(future_view.columns))}

Độ phân tán point forecast: **{forecast_spread:.2%}**. {stance_summary}. Với hybrid, `predicted_close_q10-q90` nên được ưu tiên hơn q50 khi đánh giá rủi ro.

Đây là nghiên cứu định lượng, không phải khuyến nghị đầu tư. Cần cập nhật `data.csv` để có forecast realtime.

## Biểu đồ

![VNIndex context](outputs/figures/01_vnindex_context.png)

![Model score](outputs/figures/02_model_score.png)

![Return MAE](outputs/figures/03_return_mae.png)

![Locked test forecasts](outputs/figures/04_test_forecasts_20d.png)

![Future return](outputs/figures/05_future_return_20d.png)

![Future price targets](outputs/figures/06_future_price_20d.png)

![Random Forest importance](outputs/figures/07_random_forest_feature_importance.png)

![Bootstrap score](outputs/figures/08_bootstrap_forecast_score.png)

![Future projection path](outputs/figures/09_future_projection_path.png)

![Actual vs predicted](outputs/figures/10_actual_vs_predicted.png)

![Residual diagnostics](outputs/figures/11_residual_diagnostics.png)

![Yearly stability](outputs/figures/12_yearly_score_heatmap.png)

![Hybrid future quantile band](outputs/figures/13_hybrid_future_quantile_band.png)

![Hybrid test quantiles](outputs/figures/14_hybrid_test_quantiles.png)

![OOF ensemble weights](outputs/figures/15_oof_ensemble_weights.png)

![Hybrid feature importance](outputs/figures/16_hybrid_feature_importance.png)

## Chạy lại

```bash
/home/namngyh/miniconda3/envs/eda/bin/python run_benchmark.py
```

Artifact mới gồm `hybrid_tuning_trials.csv`, `hybrid_best_hyperparameters.csv`, `quantile_metrics.csv`, `ensemble_weight_trials.csv`, `ensemble_oof_predictions.csv`, `ensemble_best_weights.csv` và `hybrid_feature_importance.csv`.

## Giới hạn

EGARCH parameters được fit trên từng training fold rồi cố định để lọc validation/test theo thời gian; HMM cũng dùng causal filtering. Tuy vậy, target 20 phiên chồng lấn làm số quan sát hiệu dụng thấp hơn số dòng. Block bootstrap/HAC giảm thiên lệch tự tin nhưng không loại bỏ model-selection bias hoặc structural breaks.
"""
    output_path.write_text(body, encoding="utf-8")
