# Phòng ngừa rò rỉ dữ liệu

Mọi split đều theo thời gian. Với từng horizon, quan sát train chỉ được dùng khi target end date nhỏ hơn boundary validation hoặc test. Không có centered rolling window, negative shift trong feature hoặc calibration fit trên test.
