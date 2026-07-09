from pathlib import Path

import pandas as pd


def _pct(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def _num(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.4f}"


def _markdown_table(df: pd.DataFrame, cols: list[str], max_rows=12) -> str:
    view = df[cols].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
    view = view.astype(str)
    widths = {
        col: max(len(col), *(len(value) for value in view[col].tolist()))
        for col in view.columns
    }
    header = "| " + " | ".join(col.ljust(widths[col]) for col in view.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in view.columns) + " |"
    rows = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def write_readme(
    output_path: Path,
    data_summary: dict,
    split_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    financial: pd.DataFrame,
    ranking: pd.DataFrame,
    artifacts: dict,
):
    best_by_horizon = (
        ranking.sort_values(["horizon", "rank_score"], ascending=[True, False])
        .groupby("horizon")
        .head(3)
        .reset_index(drop=True)
    )
    best_financial = (
        financial.sort_values(["horizon", "strategy_sharpe"], ascending=[True, False])
        .groupby("horizon")
        .head(3)
        .reset_index(drop=True)
    )
    macd = ranking[ranking["model"] == "MACD 12-26-9"].copy()

    body = f"""# VNIndex ML Forecast Benchmark vs MACD

Repo này so sánh khả năng dự báo của MACD 12-26-9 với các mô hình Machine Learning:
SVC, SVR, Random Forest, XGBoost, LightGBM, CatBoost và HMM/Regime Model.

Trọng tâm là dự báo theo 3 khung thời gian:

- Ngắn hạn: 5 phiên
- Trung hạn: 20 phiên
- Dài hạn: 60 phiên

## Dữ liệu và phương pháp

- File gốc: `data.csv`
- Số dòng hợp lệ: `{data_summary["rows"]:,}`
- Giai đoạn dữ liệu: `{data_summary["start"]}` đến `{data_summary["end"]}`
- Biến dự báo: return tương lai `close[t+h] / close[t] - 1`
- Nhãn hướng: tăng nếu return tương lai lớn hơn 0
- Chia tập: theo thời gian, không shuffle, tránh leakage
- Chiến lược tài chính: long/flat; nếu mô hình dự báo tăng thì nắm giữ cho phiên kế tiếp, nếu không thì đứng ngoài
- MACD baseline: `MACD line > Signal line` được xem là tín hiệu bullish để dự báo hướng

### Split thời gian

{_markdown_table(split_summary, ["split", "rows", "start", "end"])}

## Kết luận nhanh

Top 3 mô hình theo điểm tổng hợp dự báo gồm `balanced_accuracy`, `f1`, `spearman_ic`, `r2` và `strategy_sharpe`:

{_markdown_table(best_by_horizon, ["horizon", "model", "rank_score", "balanced_accuracy", "f1", "spearman_ic", "strategy_sharpe"])}

Top 3 mô hình theo Sharpe chiến lược:

{_markdown_table(best_financial, ["horizon", "model", "strategy_total_return", "strategy_sharpe", "strategy_max_drawdown", "strategy_exposure"])}

Vị trí của MACD trong bảng dự báo:

{_markdown_table(macd, ["horizon", "model", "rank_score", "balanced_accuracy", "f1", "spearman_ic", "strategy_sharpe"])}

## Chỉ số học máy

Các chỉ số chính:

- `accuracy`: tỷ lệ dự báo đúng hướng tăng/giảm.
- `balanced_accuracy`: accuracy cân bằng giữa lớp tăng và giảm, hữu ích khi thị trường thiên lệch tăng.
- `precision`: khi mô hình báo tăng, tỷ lệ đúng là bao nhiêu.
- `recall`: trong các giai đoạn thực tế tăng, mô hình bắt được bao nhiêu.
- `f1`: cân bằng giữa precision và recall.
- `roc_auc`: khả năng xếp hạng xác suất tăng.
- `mae`, `rmse`, `r2`: sai số dự báo return.
- `spearman_ic`: Information Coefficient dạng rank correlation giữa return dự báo và return thực tế.

Ảnh heatmap:

![Balanced Accuracy]({artifacts["balanced_accuracy_heatmap"]})

![Sharpe]({artifacts["sharpe_heatmap"]})

## Chỉ số tài chính

Các chỉ số chính:

- `strategy_total_return`: tổng lợi nhuận chiến lược long/flat trên test.
- `strategy_cagr`: tăng trưởng kép năm hóa.
- `strategy_ann_vol`: biến động năm hóa.
- `strategy_sharpe`: lợi nhuận điều chỉnh rủi ro.
- `strategy_sortino`: Sharpe chỉ phạt downside volatility.
- `strategy_max_drawdown`: mức sụt giảm lớn nhất.
- `strategy_calmar`: CAGR / Max Drawdown.
- `strategy_profit_factor`: tổng lãi / tổng lỗ.
- `strategy_exposure`: tỷ lệ thời gian ở trạng thái long.
- `strategy_turnover`: mức thay đổi vị thế bình quân.
- `strategy_beta_to_buy_hold`, `strategy_alpha_annualized`, `strategy_information_ratio`: so với buy-and-hold.

## Trực quan hóa theo mô hình và horizon

### Tổng quan giá, MACD và RSI

![Price MACD RSI]({artifacts["price_macd"]})

### Equity curves

![Equity 5d]({artifacts["equity_5"]})

![Equity 20d]({artifacts["equity_20"]})

![Equity 60d]({artifacts["equity_60"]})

### Forecast panels

Mỗi panel hiển thị return tương lai thực tế, return dự báo và vùng xanh là giai đoạn mô hình chọn long.

![Forecast 5d]({artifacts["forecast_5"]})

![Forecast 20d]({artifacts["forecast_20"]})

![Forecast 60d]({artifacts["forecast_60"]})

### Feature importance

![Feature Importance]({artifacts["feature_importance"]})

## Cách chạy lại

```bash
/home/namngyh/miniconda3/envs/eda/bin/python run_benchmark.py
```

Kết quả được ghi vào `outputs/`:

- `metrics_by_horizon.csv`: chỉ số học máy theo mô hình và horizon.
- `financial_metrics_by_horizon.csv`: chỉ số tài chính theo mô hình và horizon.
- `model_ranking.csv`: bảng xếp hạng tổng hợp.
- `predictions.csv`: dự báo từng ngày trên tập test.
- `feature_importance.csv`: top feature importance của các mô hình cây/boosting.
- `regime_summary.csv`: trạng thái HMM và return kỳ vọng theo regime.
- `figures/*.png`: toàn bộ biểu đồ.

## Lưu ý diễn giải

Kết quả này là out-of-sample theo split thời gian, nhưng vẫn là nghiên cứu lịch sử. Nếu dùng giao dịch thật cần bổ sung transaction cost, slippage, walk-forward retraining, kiểm định ổn định theo từng giai đoạn thị trường và quản trị rủi ro vị thế.
"""
    output_path.write_text(body, encoding="utf-8")
