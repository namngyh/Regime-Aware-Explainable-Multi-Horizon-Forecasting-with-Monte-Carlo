# VNIndex 20-session Forecast: Optimized MACD and HMM

Project đã được rút gọn còn hai mô hình tốt nhất cho dự báo 20 phiên: **MACD 8-24-9** và **HMM Regime 4 trạng thái (`tied` covariance)**. SVC, SVR, Random Forest, XGBoost, LightGBM và CatBoost đã bị loại khỏi pipeline và dependency.

## Kết quả lựa chọn trên tập test khóa

| rank | model       | forecast_score | mae    | rmse   | price_mae | price_rmse | price_mape | balanced_accuracy | spearman_ic |
| ---- | ----------- | -------------- | ------ | ------ | --------- | ---------- | ---------- | ----------------- | ----------- |
| 1    | MACD 8-24-9 | 0.0788         | 0.0411 | 0.0537 | 54.9762   | 74.4659    | 0.0409     | 0.5370            | 0.0463      |
| 2    | HMM Regime  | 0.0373         | 0.0432 | 0.0549 | 58.0273   | 76.5673    | 0.0429     | 0.4698            | 0.0157      |

`forecast_score` ưu tiên đúng theo yêu cầu:

- 65% chất lượng dự báo lợi suất: MAE skill 35% và RMSE skill 30%.
- 25% chất lượng dự báo giá đóng cửa: MAE skill 15% và RMSE skill 10%.
- 10% khả năng dự báo đúng hướng: Balanced Accuracy.

Các skill score so sánh với dự báo không đổi (`return = 0`, giá tương lai bằng giá hiện tại). Điểm cao hơn tốt hơn; MAE/RMSE thấp hơn tốt hơn. Tập test không tham gia chọn hyperparameter.

## Hyperparameter tối ưu

| model      | candidate_id | cv_score | cv_score_std | selected_params                                               |
| ---------- | ------------ | -------- | ------------ | ------------------------------------------------------------- |
| MACD       | 3            | 0.0392   | 0.0152       | {"fast": 8, "signal": 9, "slow": 24}                          |
| HMM Regime | 3            | 0.0490   | 0.0166       | {"covariance_type": "tied", "n_components": 4, "n_iter": 400} |

- Search: 14 cấu hình, 3-fold expanding `TimeSeriesSplit`, `gap=20`.
- Tổng thời gian fit cộng dồn: 5.7 giây.
- Horizon cố định: 20 phiên giao dịch.

## Chia dữ liệu theo thời gian

| split | rows | start      | end        |
| ----- | ---- | ---------- | ---------- |
| train | 4295 | 2001-05-18 | 2019-01-11 |
| valid | 920  | 2019-01-14 | 2022-09-20 |
| test  | 921  | 2022-09-21 | 2026-06-03 |

Dữ liệu gốc có 6,298 quan sát từ 2000-07-28 đến 2026-07-01. Mọi feature kỹ thuật đều chỉ dùng thông tin tại hoặc trước thời điểm dự báo.

## Độ ổn định theo năm trên tập test

| model       | year | rows | forecast_score | mae    | rmse   | price_mae | balanced_accuracy |
| ----------- | ---- | ---- | -------------- | ------ | ------ | --------- | ----------------- |
| HMM Regime  | 2022 | 73   | 0.0728         | 0.0669 | 0.0788 | 69.7182   | 0.6393            |
| HMM Regime  | 2023 | 249  | 0.0670         | 0.0345 | 0.0412 | 38.7818   | 0.5025            |
| HMM Regime  | 2024 | 250  | -0.0550        | 0.0301 | 0.0362 | 37.4625   | 0.4088            |
| HMM Regime  | 2025 | 249  | 0.0906         | 0.0511 | 0.0639 | 73.8768   | 0.4092            |
| HMM Regime  | 2026 | 100  | -0.0806        | 0.0611 | 0.0742 | 109.3611  | 0.2677            |
| MACD 8-24-9 | 2022 | 73   | 0.0478         | 0.0683 | 0.0799 | 71.4171   | 0.5619            |
| MACD 8-24-9 | 2023 | 249  | 0.0635         | 0.0339 | 0.0422 | 38.2682   | 0.4885            |
| MACD 8-24-9 | 2024 | 250  | 0.0466         | 0.0268 | 0.0340 | 33.2723   | 0.5323            |
| MACD 8-24-9 | 2025 | 249  | 0.1306         | 0.0493 | 0.0628 | 71.3766   | 0.5890            |
| MACD 8-24-9 | 2026 | 100  | 0.0355         | 0.0548 | 0.0686 | 98.0005   | 0.5101            |

## Dự báo từ phiên mới nhất

VNIndex gần nhất trong dữ liệu đóng cửa ở **1,865.37**.

| as_of_date | target_date | model       | pred_return | predicted_close | test_forecast_score |
| ---------- | ----------- | ----------- | ----------- | --------------- | ------------------- |
| 2026-07-01 | 2026-07-29  | MACD 8-24-9 | 1.47%       | 1892.8329       | 0.0788              |
| 2026-07-01 | 2026-07-29  | HMM Regime  | 5.11%       | 1960.6776       | 0.0373              |

Đây là dự báo nghiên cứu từ dữ liệu lịch sử, không phải khuyến nghị đầu tư. Vì dữ liệu kết thúc ở `2026-07-01`, cần cập nhật `data.csv` và chạy lại pipeline nếu muốn tín hiệu mới hơn.

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
