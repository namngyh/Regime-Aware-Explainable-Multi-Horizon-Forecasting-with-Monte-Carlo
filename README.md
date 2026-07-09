# VNIndex ML Forecast Benchmark vs MACD

Repo này so sánh khả năng dự báo của MACD 12-26-9 với các mô hình Machine Learning:
SVC, SVR, Random Forest, XGBoost, LightGBM, CatBoost và HMM/Regime Model.

Trọng tâm là dự báo theo 3 khung thời gian:

- Ngắn hạn: 5 phiên
- Trung hạn: 20 phiên
- Dài hạn: 60 phiên

## Dữ liệu và phương pháp

- File gốc: `data.csv`
- Số dòng hợp lệ: `6,298`
- Giai đoạn dữ liệu: `2000-07-28` đến `2026-07-01`
- Biến dự báo: return tương lai `close[t+h] / close[t] - 1`
- Nhãn hướng: tăng nếu return tương lai lớn hơn 0
- Chia tập: theo thời gian, không shuffle, tránh leakage
- Chiến lược tài chính: long/flat; nếu mô hình dự báo tăng thì nắm giữ cho phiên kế tiếp, nếu không thì đứng ngoài
- MACD baseline: `MACD line > Signal line` được xem là tín hiệu bullish để dự báo hướng

### Split thời gian

| split | rows | start      | end        |
| ----- | ---- | ---------- | ---------- |
| train | 4267 | 2001-05-18 | 2018-11-30 |
| valid | 914  | 2018-12-03 | 2022-08-01 |
| test  | 915  | 2022-08-02 | 2026-04-03 |

## Kết luận nhanh

Top 3 mô hình theo điểm tổng hợp dự báo gồm `balanced_accuracy`, `f1`, `spearman_ic`, `r2` và `strategy_sharpe`:

| horizon | model         | rank_score | balanced_accuracy | f1     | spearman_ic | strategy_sharpe |
| ------- | ------------- | ---------- | ----------------- | ------ | ----------- | --------------- |
| 5       | HMM Regime    | 0.8000     | 0.5357            | 0.6066 | 0.0479      | 1.5148          |
| 5       | SVC           | 0.7750     | 0.5217            | 0.6531 | 0.0364      | 0.9470          |
| 5       | MACD 12-26-9  | 0.7000     | 0.5184            | 0.5725 | 0.0254      | 0.9119          |
| 20      | MACD 12-26-9  | 0.8000     | 0.5329            | 0.5966 | 0.0338      | 0.9119          |
| 20      | XGBoost       | 0.7250     | 0.5344            | 0.6650 | 0.0137      | 0.2267          |
| 20      | Random Forest | 0.6000     | 0.5049            | 0.6469 | -0.1057     | 0.5679          |
| 60      | MACD 12-26-9  | 0.7000     | 0.4658            | 0.5714 | 0.0197      | 0.9119          |
| 60      | HMM Regime    | 0.7000     | 0.4392            | 0.6852 | -0.0911     | 0.9145          |
| 60      | Random Forest | 0.6500     | 0.5115            | 0.6554 | -0.1179     | 0.4529          |

Top 3 mô hình theo Sharpe chiến lược:

| horizon | model        | strategy_total_return | strategy_sharpe | strategy_max_drawdown | strategy_exposure |
| ------- | ------------ | --------------------- | --------------- | --------------------- | ----------------- |
| 5       | HMM Regime   | 0.8485                | 1.5148          | -0.0943               | 0.5727            |
| 5       | SVC          | 0.5795                | 0.9470          | -0.1663               | 0.7049            |
| 5       | MACD 12-26-9 | 0.4274                | 0.9119          | -0.1779               | 0.5344            |
| 20      | HMM Regime   | 0.8485                | 1.5148          | -0.0943               | 0.5727            |
| 20      | MACD 12-26-9 | 0.4274                | 0.9119          | -0.1779               | 0.5344            |
| 20      | SVC          | 0.5281                | 0.9002          | -0.1837               | 0.5880            |
| 60      | HMM Regime   | 0.5408                | 0.9145          | -0.1291               | 0.7770            |
| 60      | MACD 12-26-9 | 0.4274                | 0.9119          | -0.1779               | 0.5344            |
| 60      | LightGBM     | 0.4255                | 0.7960          | -0.1384               | 0.5005            |

Vị trí của MACD trong bảng dự báo:

| horizon | model        | rank_score | balanced_accuracy | f1     | spearman_ic | strategy_sharpe |
| ------- | ------------ | ---------- | ----------------- | ------ | ----------- | --------------- |
| 5       | MACD 12-26-9 | 0.7000     | 0.5184            | 0.5725 | 0.0254      | 0.9119          |
| 20      | MACD 12-26-9 | 0.8000     | 0.5329            | 0.5966 | 0.0338      | 0.9119          |
| 60      | MACD 12-26-9 | 0.7000     | 0.4658            | 0.5714 | 0.0197      | 0.9119          |

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

![Balanced Accuracy](outputs/figures/02_balanced_accuracy_heatmap.png)

![Sharpe](outputs/figures/03_strategy_sharpe_heatmap.png)

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

![Price MACD RSI](outputs/figures/01_price_macd_rsi.png)

### Equity curves

![Equity 5d](outputs/figures/04_equity_curves_5d.png)

![Equity 20d](outputs/figures/04_equity_curves_20d.png)

![Equity 60d](outputs/figures/04_equity_curves_60d.png)

### Forecast panels

Mỗi panel hiển thị return tương lai thực tế, return dự báo và vùng xanh là giai đoạn mô hình chọn long.

![Forecast 5d](outputs/figures/05_forecast_panel_5d.png)

![Forecast 20d](outputs/figures/05_forecast_panel_20d.png)

![Forecast 60d](outputs/figures/05_forecast_panel_60d.png)

### Feature importance

![Feature Importance](outputs/figures/06_feature_importance.png)

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
