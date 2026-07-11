# Phương pháp RAEMF-MC

RAEMF-MC dự báo riêng cho 20, 40 và 60 phiên. Target dùng forward log return
chuẩn hóa bởi volatility có sẵn tại ngày dự báo. `Stress` ghi đè nhãn hướng nếu
maximum adverse excursion thấp hơn `-1.5 × sigma_t × sqrt(h)`.

HMM chỉ nhận diện regime bằng forward-filtered probability. EGARCH Student-t đo
volatility/tail risk và được so sánh với GJR-GARCH/GARCH bằng QLIKE, không dùng
độ chính xác return để chọn. EBM multiclass dự báo bốn trạng thái; sigmoid hoặc
isotonic calibration chỉ học trên validation/OOF. Moving Block Bootstrap giữ
phụ thuộc chuỗi thời gian. Monte Carlo mô phỏng transition HMM, innovation đuôi
dày và volatility theo regime; chế độ reweighted điều chỉnh trọng số mềm theo
xác suất EBM, không ép nhãn path.

