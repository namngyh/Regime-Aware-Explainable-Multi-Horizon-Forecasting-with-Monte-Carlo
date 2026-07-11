# Phòng ngừa leakage

- Mỗi horizon có `target_end_date_h` là ngày giao dịch t+h.
- Train chỉ nhận dòng có target end trước validation; train-validation chỉ nhận
  dòng có target end trước test.
- Feature là backward-looking; MAE/MFE/forward return chỉ là target.
- Scaler và HMM fit trên train. HMM dùng `P(S_t|F_t)`, không smoothing.
- Calibration học trên validation trước test.
- Actual future path chỉ dùng đánh giá fan-chart coverage.
- Signal sau close t chỉ áp dụng cho return t+1 trong backtest.

