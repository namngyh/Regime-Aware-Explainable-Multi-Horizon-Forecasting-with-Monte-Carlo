# Báo cáo kỹ thuật

## Tóm tắt

Dự án xây dựng RAEMF-MC cho VN-Index với ba chân trời 20, 40 và 60 phiên. Nhãn được xây dựng từ lợi suất tương lai chuẩn hóa bởi biến động nhân quả và có lớp Stress dựa trên maximum adverse excursion trong tương lai.

## Bối cảnh và động cơ nghiên cứu

Thị trường có thể chuyển trạng thái giữa tăng, đi ngang, giảm và stress. Vì vậy, mô hình xác suất đa lớp có khả năng trình bày bất định phù hợp hơn một dự báo điểm đơn lẻ.

## Kiến trúc RAEMF-MC

Pipeline gồm kiểm tra dữ liệu, đặc trưng nhân quả, Filtered HMM, EGARCH Student-t, EBM đa chân trời, calibration, moving block bootstrap, Monte Carlo có điều kiện trạng thái và backtest exposure minh họa.

## Phòng ngừa leakage

Tập train, validation và test được chia theo thời gian. Các quan sát có nhãn kết thúc vượt boundary bị loại khỏi phần trước boundary. Tập test chỉ dùng sau khi mô hình và calibration đã cố định.

## Hạn chế

Không có dữ liệu vĩ mô, market breadth hoặc thành phần chỉ số. Backtest exposure không đại diện cho khả năng giao dịch thực tế của VN-Index.
