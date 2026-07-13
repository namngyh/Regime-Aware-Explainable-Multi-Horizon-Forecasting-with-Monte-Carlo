# Hướng dẫn cho tác nhân phát triển

- Không dùng dữ liệu tương lai khi tạo đặc trưng, chọn tham số, hiệu chỉnh xác suất hoặc backtest.
- Với mỗi horizon, train và validation phải được purge bằng `target_end_date_h < boundary`.
- Không dùng test để tuning, chọn feature, chọn calibration hoặc chọn ngưỡng.
- Không bịa số liệu. Thành phần không ước lượng được phải ghi rõ lý do.
- Báo cáo chính bằng tiếng Việt, trung lập, không đưa lời khuyên đầu tư.
- Lệnh kiểm thử chính: `conda run -n eda python -m pytest -q`.
- Lệnh chạy laptop: `bash scripts/run_laptop.sh`.
