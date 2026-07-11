# RAEMF-MC

**Regime-Aware Explainable Multi-Horizon Forecasting with Monte Carlo**
Mô hình dự báo đa chân trời có giải thích, nhận biết trạng thái thị trường và mô
phỏng Monte Carlo.

RAEMF-MC ước lượng xác suất **Bull, Sideway, Bear và Stress** của VN-Index trong
20, 40 và 60 phiên. Hệ thống đồng thời nhận diện regime hiện tại, dự báo
volatility, mô phỏng nhiều quỹ đạo tương lai, lượng hóa bất định và chuyển kết
quả thành market filter Risk-on/Neutral/Risk-off/Uncertain.

> Đây là nghiên cứu định lượng, không phải lời khuyên đầu tư. Xác suất không phải
> cam kết, fan chart không phải quỹ đạo chắc chắn và backtest không bảo đảm lợi
> nhuận tương lai.

## Mục tiêu và phạm vi

Hệ thống trả lời: thị trường hiện ở regime nào; 1–3 tháng tới nghiêng về trạng
thái nào; mức volatility/drawdown risk ra sao; dự báo tự tin đến đâu; yếu tố nào
đóng góp; và market filter có giá trị tăng thêm so với MACD không.

Hệ thống không cam kết điểm số tương lai, không đưa lệnh thật, không thay thế
quyết định đầu tư và không phải chiến lược autotrading hoàn chỉnh.

## Pipeline

```text
OHLCV -> causal features -> purged split -------------------------+
                    |-> Filtered HMM: "thời tiết" hiện tại         |
                    |-> EGARCH-t: "mức độ giông bão"               |-> EBM 20/40/60
                    +-> technical / volume ------------------------+      |
Moving Block Bootstrap -> độ ổn định                                     +-> calibration
HMM transition + tail innovations -> Monte Carlo scenarios                |
                                      confidence + drift + disagreement -> market filter
```

- **Filtered HMM** dùng xác suất lọc `P(S_t|F_t)`, không dùng thông tin tương lai.
- **EGARCH Student-t** đo volatility bất đối xứng và rủi ro đuôi dày; GJR-GARCH-t
  và GARCH-t là challenger.
- **EBM** là mô hình multiclass chính, huấn luyện riêng cho 20/40/60 phiên.
- **Calibration** học trên validation/OOF, tuyệt đối không fit trên test.
- **Block Bootstrap** resample block liên tiếp; không IID-row bootstrap.
- **Monte Carlo** có structural mode và soft EBM reweighting.

## Target và chống leakage

Với horizon `h`, `r(t,h)=log(P(t+h)/P(t))` và
`z(t,h)=r(t,h)/(sigma(t)*sqrt(h)+epsilon)`. Bull khi `z>0.5`, Bear khi `z<-0.5`,
còn lại Sideway. Stress ghi đè nếu future maximum adverse excursion nhỏ hơn
`-1.5*sigma(t)*sqrt(h)`.

Mỗi dòng có `target_end_date_20/40/60`. Train chỉ chứa nhãn kết thúc trước ngày
validation; train-validation chỉ chứa nhãn kết thúc trước ngày test. Chi tiết:
[architecture review](docs/architecture_review.md) và
[leakage prevention](docs/leakage_prevention.md).

## Cách đọc đầu ra mới nhất

Ví dụ “Bull 61%, Stress 6%, confidence Medium” nghĩa là mô hình nghiêng tăng,
nhưng chưa đủ chắc chắn cho exposure tối đa. Hai dự báo cùng Bull 65% có thể có
độ tin cậy khác nhau nếu Bootstrap interval, HMM entropy hoặc feature drift khác
nhau. Fan chart rộng nghĩa là nhiều kịch bản còn hợp lý; fan hẹp không có nghĩa
dự báo chắc chắn đúng.

Báo cáo hiện tại ở [market outlook](outputs/latest/market_outlook.md). Hướng dẫn
phổ thông ở [investor guide](docs/investor_guide_vi.md).

## Kết quả quick-mode hiện tại

Holdout test nhân quả cho thấy RAEMF-MC có Brier/log loss tốt hơn baseline MACD
rule và accuracy tổng thể cao hơn. Tuy nhiên Bear/Stress recall của lần chạy
quick-mode hiện tại bằng 0 ở nhiều horizon; do đó **chưa đủ bằng chứng kết luận
full model tốt hơn MACD** theo tiêu chuẩn nghiên cứu. Trạng thái mới nhất được
gắn `Uncertain`, thay vì ép mô hình đưa ra kết luận mạnh.

| Horizon | Model | Balanced Accuracy | Macro F1 | Brier | Log loss | Bear recall | Stress recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | RAEMF-MC | 0.272 | 0.211 | 0.684 | 1.261 | 0.000 | 0.000 |
| 20 | MACD rule | 0.238 | 0.205 | 1.175 | 2.190 | 0.323 | 0.015 |
| 40 | RAEMF-MC | 0.250 | 0.204 | 0.673 | 1.218 | 0.000 | 0.000 |
| 40 | MACD rule | 0.229 | 0.184 | 1.207 | 2.247 | 0.293 | 0.020 |
| 60 | RAEMF-MC | 0.239 | 0.157 | 0.682 | 1.276 | 0.000 | 0.000 |
| 60 | MACD rule | 0.234 | 0.185 | 1.195 | 2.226 | 0.298 | 0.007 |

Quick-mode dùng một outer holdout và Moving Block Bootstrap trên fixed causal
predictions cho diagnostics. Nó không được trình bày như full model-refit
Bootstrap. Các artifact chưa ước lượng trong quick-mode ghi rõ `not estimated`,
không được điền số giả.

## Cài đặt và chạy

```bash
conda activate eda
pip install -r requirements.txt
python -m pytest -q
python -m scripts.run_regime_pipeline \
  --data data.csv --config config/model_config.yaml \
  --output outputs/runs --quick-mode
python -m scripts.generate_reports --run-dir outputs/runs/<run_id> --data data.csv
```

Điều chỉnh seed, horizons, số Monte Carlo paths, block length và threshold trong
`config/model_config.yaml`. Mỗi run lưu config snapshot, Git commit, data SHA-256,
seed, feature list, dates, warnings, metrics, prediction và plots trong
`outputs/runs/timestamp_gitcommit/`.

## Cấu trúc chính

- `src/targets.py`, `src/validation.py`: target và purge/leakage assertions.
- `src/features.py`, `src/feature_*`: feature, registry, redundancy và drift.
- `src/regime/`: Filtered HMM, alignment và economic interpretation.
- `src/risk/`: EGARCH/GJR/GARCH Student-t và volatility diagnostics.
- `src/ebm_multihorizon.py`, `src/calibration.py`: EBM và calibration.
- `src/simulation/`: block bootstrap và Monte Carlo.
- `src/market_filter.py`, `src/uncertainty.py`: confidence/exposure.
- `src/baselines/`: MACD benchmark.
- `scripts/`: CLI; `tests/`: test chống leakage và tái lập.
- `docs/model_card.md`, `docs/methodology.md`: tài liệu nghiên cứu.

## Output

Mỗi run xuất prediction/calibrated probabilities, classification/probability/
calibration metrics, HMM diagnostics/transition/alignment, EGARCH diagnostics,
volatility forecast, Monte Carlo summary/path sample/coverage, feature
importance/drift, market state, uncertainty, MACD comparison, local explanation,
reproducibility metadata và report Markdown. Các file legacy ở `outputs/` được
giữ làm benchmark lịch sử và không lẫn với run RAEMF-MC.

## Hạn chế

Dữ liệu chủ yếu là giá/khối lượng; breadth và macro chưa có. Target chồng lấn,
structural break, HMM state ẩn, calibration suy giảm, giả định Bootstrap/Monte
Carlo và sự kiện cực đoan chưa từng thấy đều có thể làm kết quả ngoài mẫu xấu đi.
EBM là mô hình lai có khả năng giải thích, không phải “hộp trắng tuyệt đối”. Không
có ngôn ngữ nhân quả: feature chỉ đóng góp vào xác suất mô hình ước lượng.
