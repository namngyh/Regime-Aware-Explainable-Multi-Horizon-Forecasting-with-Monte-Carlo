# Rà soát kiến trúc RAEMF-MC

## Phạm vi và trạng thái ban đầu

Repository ban đầu là benchmark dự báo lợi suất VN-Index 20 phiên. Luồng chính là
`data.csv -> src/data.py -> src/features.py -> src/benchmark.py -> outputs/`.
Các mô hình hiện hữu gồm HMM, SVR, Random Forest, HMM-RF ensemble và
HMM-EGARCH-LightGBM quantile. Chúng được giữ nguyên làm benchmark; pipeline mới
được bổ sung theo module và không phá API cũ.

## Dữ liệu

`data.csv` chứa Date, Open, High, Low, Close và Volume từ 28/07/2000 đến
01/07/2026. File có dấu phẩy phân tách hàng nghìn làm số cột thay đổi; bộ đọc
`src/data.py` xử lý riêng định dạng này, sắp xếp thời gian và loại ngày trùng.
Không có breadth, macro hay dữ liệu thành phần chỉ số; hệ thống không giả lập các
nguồn này.

## Kiến trúc cũ và thành phần được giữ

- `src/data.py`: bộ đọc dữ liệu được tái sử dụng.
- `src/features.py`: các feature kỹ thuật nhân quả được giữ sau khi bỏ biến trùng.
- `src/models.py`: HMM/SVR/Random Forest là benchmark.
- `src/hybrid.py`: LightGBM quantile và EGARCH feature là benchmark/challenger.
- `src/metrics.py`, `src/plots.py`, `src/research.py`: tái sử dụng hàm phù hợp.
- Toàn bộ output cũ được giữ trong `outputs/` để truy vết, nhưng không được xem
  là bằng chứng của RAEMF-MC mới.

## Vấn đề phát hiện

1. Split 70/15/15 cũ không purge outer boundary. Nhãn của các dòng cuối train có
   thể dùng giá nằm trong validation; nhãn cuối train-validation có thể dùng giá
   test.
2. `TimeSeriesSplit(gap=horizon)` chỉ giảm overlap ở inner CV, không chứng minh
   outer test purge.
3. `ret_n` và `mom_n` giống hệt nhau về toán học.
4. Target cũ chỉ là return/up ở horizon 20; chưa có 40/60 và chưa nhận diện
   Stress theo đường đi nội kỳ.
5. HMM đã forward-filter khi dự báo nhưng chưa có multi-seed selection, state
   alignment, entropy, duration và ổn định qua fold.
6. EGARCH được fit trên train và fixed-filter trên chuỗi sau đó, nhưng chưa có
   quy trình lựa chọn bằng QLIKE/coverage, multi-step simulation hay residual
   diagnostics đầy đủ.
7. LightGBM quantile dự báo return, không phải EBM multiclass; xác suất suy ra
   chưa calibration.
8. Chưa có block-bootstrap probability, structural/reweighted Monte Carlo,
   Uncertain state, market filter hoặc backtest lag t+1.
9. Tuning/report cũ phụ thuộc một custom score; chưa đủ metric xác suất, lớp
   Stress và kiểm định chênh lệch với MACD.
10. Output đặt chung một thư mục, thiếu config snapshot, checksum, package list,
    warning và run id.

## Kiến trúc đích

```text
OHLCV -> feature nhân quả -> target + target_end_date
      -> purged train/validation/test
      -> Filtered HMM probabilities ----+
      -> EGARCH Student-t risk ---------+-> EBM_20/40/60 -> calibration
      -> technical/volume features -----+       |
                                                +-> uncertainty/market filter
train only -> moving block bootstrap -----------+
HMM + EGARCH + Student-t -> Monte Carlo --------+
                                                -> reports/plots/backtest
```

Scaler, HMM, EGARCH, EBM, calibration và mọi threshold đều được fit/chọn chỉ từ
phần dữ liệu được phép ở thời điểm dự báo. Actual future path chỉ dùng đánh giá.

## Migration

1. Thêm `target_end_date_h`, purged split và test chống overlap.
2. Tạo target Bull/Sideway/Bear/Stress riêng cho 20/40/60.
3. Mở rộng feature nhân quả, registry/availability và drift; breadth là optional.
4. Dùng Filtered HMM làm context, align state bằng Hungarian assignment.
5. Dùng EGARCH Student-t làm risk model và EBM multiclass làm forecaster chính.
6. Calibration từ validation/OOF; test chỉ đánh giá một lần.
7. Block bootstrap và Monte Carlo tách epistemic/aleatoric uncertainty.
8. Market filter lag một phiên; benchmark/ablation bắt buộc gồm MACD.
9. Mỗi run có thư mục riêng, config/checksum/version/seed và output chuẩn hóa.

## Tái lập và output cũ

Seed nằm trong YAML; dữ liệu được hash; package version và Git commit được lưu
theo run. Các file output cũ không bị xóa nhưng README mới phân biệt rõ legacy
benchmark với RAEMF-MC. Đường dẫn CLI là tương đối, không hard-code máy cá nhân.

