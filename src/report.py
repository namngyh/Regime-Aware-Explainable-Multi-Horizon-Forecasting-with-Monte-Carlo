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
        column: max(
            len(column), *(len(value) for value in view[column].tolist())
        )
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
        + " | ".join(
            row[column].ljust(widths[column]) for column in view.columns
        )
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
            "price_mae",
            "price_rmse",
            "price_mape",
            "balanced_accuracy",
            "spearman_ic",
        ]
    ]
    parameter_view = best_parameters[
        [
            "model",
            "candidate_id",
            "cv_score",
            "cv_score_std",
            "selected_params",
        ]
    ]
    future_view = future[
        [
            "as_of_date",
            "target_date",
            "model",
            "pred_return",
            "predicted_close",
            "test_forecast_score",
        ]
    ].copy()
    future_view["pred_return"] = future_view["pred_return"].map(
        lambda value: f"{value:.2%}"
    )
    latest_close = future["latest_close"].iloc[0]
    total_seconds = tuning_trials["fit_seconds"].sum()

    body = f"""# VNIndex 20-session Forecast: Optimized MACD and HMM

Project đã được rút gọn còn hai mô hình tốt nhất cho dự báo 20 phiên: **MACD 8-24-9** và **HMM Regime 4 trạng thái (`tied` covariance)**. SVC, SVR, Random Forest, XGBoost, LightGBM và CatBoost đã bị loại khỏi pipeline và dependency.

## Kết quả lựa chọn trên tập test khóa

{_table(ranking_view, list(ranking_view.columns))}

`forecast_score` ưu tiên đúng theo yêu cầu:

- 65% chất lượng dự báo lợi suất: MAE skill 35% và RMSE skill 30%.
- 25% chất lượng dự báo giá đóng cửa: MAE skill 15% và RMSE skill 10%.
- 10% khả năng dự báo đúng hướng: Balanced Accuracy.

Các skill score so sánh với dự báo không đổi (`return = 0`, giá tương lai bằng giá hiện tại). Điểm cao hơn tốt hơn; MAE/RMSE thấp hơn tốt hơn. Tập test không tham gia chọn hyperparameter.

## Hyperparameter tối ưu

{_table(parameter_view, list(parameter_view.columns))}

- Search: {len(tuning_trials)} cấu hình, 3-fold expanding `TimeSeriesSplit`, `gap=20`.
- Tổng thời gian fit cộng dồn: {total_seconds:.1f} giây.
- Horizon cố định: 20 phiên giao dịch.

## Chia dữ liệu theo thời gian

{_table(split_summary, ["split", "rows", "start", "end"])}

Dữ liệu gốc có {data_summary['rows']:,} quan sát từ {data_summary['start']} đến {data_summary['end']}. Mọi feature kỹ thuật đều chỉ dùng thông tin tại hoặc trước thời điểm dự báo.

## Độ ổn định theo năm trên tập test

{_table(stability, ["model", "year", "rows", "forecast_score", "mae", "rmse", "price_mae", "balanced_accuracy"])}

## Dự báo từ phiên mới nhất

VNIndex gần nhất trong dữ liệu đóng cửa ở **{latest_close:,.2f}**.

{_table(future_view, list(future_view.columns))}

Đây là dự báo nghiên cứu từ dữ liệu lịch sử, không phải khuyến nghị đầu tư. Vì dữ liệu kết thúc ở `{future['as_of_date'].iloc[0]}`, cần cập nhật `data.csv` và chạy lại pipeline nếu muốn tín hiệu mới hơn.

## Biểu đồ

![VNIndex context](outputs/figures/01_vnindex_context.png)

![Model score](outputs/figures/02_model_score.png)

![Return MAE](outputs/figures/03_return_mae.png)

![Locked test forecasts](outputs/figures/04_test_forecasts_20d.png)

![Future return](outputs/figures/05_future_return_20d.png)

![Future price](outputs/figures/06_future_price_20d.png)

## Chạy lại

```bash
/home/namngyh/miniconda3/envs/eda/bin/python run_benchmark.py
```

Các artifact chính nằm trong `outputs/`:

- `model_ranking.csv`: xếp hạng hai mô hình được giữ lại.
- `forecast_metrics.csv`: sai số lợi suất, giá và chỉ số hướng.
- `predictions.csv`: dự báo từng quan sát trên test.
- `test_stability_by_year.csv`: kiểm tra độ ổn định theo năm.
- `best_hyperparameters.csv`, `tuning_trials.csv`: tuning có thể tái lập cho hai mô hình.
- `future_forecasts.csv`: dự báo 20 phiên từ dòng dữ liệu mới nhất.
- `model_selection_audit.csv`, `model_selection_parameters_audit.csv`: bằng chứng so sánh tám mô hình trước khi dọn dẹp.

## Phạm vi sau dọn dẹp

Pipeline hiện chỉ hỗ trợ horizon 20 và hai mô hình chiến thắng. Việc cố chạy horizon khác sẽ báo lỗi rõ ràng. Các thư viện XGBoost, LightGBM và CatBoost không còn là dependency của project.
"""
    output_path.write_text(body, encoding="utf-8")
