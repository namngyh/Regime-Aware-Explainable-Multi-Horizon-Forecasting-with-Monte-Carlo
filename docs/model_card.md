# Model card – RAEMF-MC v0.2

Mục tiêu là dự báo xác suất Bull/Sideway/Bear/Stress và rủi ro VN-Index trong
20/40/60 phiên để tham khảo như market filter. Không dùng để cam kết mức giá,
đưa lệnh thật hoặc bảo đảm lợi nhuận.

Dữ liệu hiện tại chỉ gồm OHLCV VN-Index. Validation theo thời gian và purge theo
target end date. Xác suất EBM được calibration trên validation. Filtered HMM là
regime context; EGARCH Student-t là risk model; Monte Carlo là phân phối kịch
bản. Structural break, target chồng lấn, state ẩn, dữ liệu breadth/macro thiếu và
giả định đuôi/phụ thuộc lịch sử là các hạn chế chính. Refit cần được kích hoạt
khi feature drift hoặc calibration suy giảm. Đây không phải lời khuyên đầu tư.

