# Triển vọng VN-Index mới nhất

Ngày dữ liệu: **2026-07-01**. VN-Index: **1,865.37**.

## Dành cho nhà đầu tư phổ thông

- 20 phiên: Bull 36.5%, Sideway 33.5%, Bear 10.2%, Stress 19.8%.

- 40 phiên: Bull 42.2%, Sideway 34.0%, Bear 11.0%, Stress 12.9%.

- 60 phiên: Bull 45.0%, Sideway 28.7%, Bear 4.4%, Stress 21.9%.

Độ tin cậy: **Uncertain**. Market filter: **Uncertain**, exposure tham khảo 14%.

Xác suất là mức độ nghiêng của mô hình, không phải cam kết. Monte Carlo mô tả kịch bản theo giả định lịch sử và có thể bỏ sót cú sốc mới.

## Dành cho người chuyên môn

Git commit: `85d8885`. Probability đã sigmoid-calibrate trên validation tách biệt; test không dùng để fit calibration. HMM dùng forward filtering. EGARCH được chọn theo QLIKE proxy, không theo return score.

> Không phải lời khuyên đầu tư.


## Ghi chú phương pháp
Đây là quick-mode causal holdout; các bảng ghi `not estimated` không được diễn giải như kết quả nghiên cứu đầy đủ.
