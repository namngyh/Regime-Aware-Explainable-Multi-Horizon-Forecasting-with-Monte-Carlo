"""Vietnamese report builders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raemf_mc.reporting.tables import markdown_table


def build_run_report(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    metrics = pd.read_csv(run_dir / "metrics_by_model_horizon.csv")
    latest = json.loads((run_dir / "latest_outlook.json").read_text(encoding="utf-8"))
    bootstrap = pd.read_csv(run_dir / "bootstrap_differences.csv")
    best = metrics.sort_values(["horizon", "brier"]).groupby("horizon").head(1)[["horizon", "model", "brier", "macro_f1", "balanced_accuracy"]]
    text = [
        "# Báo cáo thực nghiệm RAEMF-MC",
        "",
        "## Tóm tắt",
        "Báo cáo này ghi lại một lần chạy laptop mode trên dữ liệu `data.csv`. Kết quả là ngoài mẫu theo thứ tự thời gian và không dùng tập test để lựa chọn tham số.",
        "",
        "## Mô hình tốt nhất theo Brier score",
        markdown_table(best),
        "",
        "## Metric chính",
        markdown_table(metrics[["model", "horizon", "macro_f1", "balanced_accuracy", "brier", "log_loss", "recall_bear", "recall_stress"]], max_rows=60),
        "",
        "## Bootstrap chênh lệch metric",
        "Chênh lệch âm của Brier hoặc log loss nghĩa là RAEMF-MC thấp hơn benchmark trên metric mất mát đó.",
        markdown_table(bootstrap, max_rows=60),
        "",
        "## Dự báo mới nhất",
        "```json",
        json.dumps(latest, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Diễn giải trung lập",
        "Nếu khoảng tin cậy bootstrap của chênh lệch metric còn chứa 0, báo cáo không kết luận RAEMF-MC vượt trội ổn định. Recall của Bear và Stress được trình bày riêng vì mất cân bằng lớp có thể làm accuracy đánh lừa.",
        "",
        "## Giới hạn",
        "Dữ liệu chỉ gồm lịch sử VN-Index trong `data.csv`, không có biến vĩ mô, market breadth hoặc dữ liệu thành phần chỉ số. Backtest là minh họa exposure sau một ngày trễ, không phải khuyến nghị đầu tư.",
    ]
    (run_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def build_docs_and_readme(run_dir: str | Path | None = None) -> None:
    run_dir = Path(run_dir) if run_dir else None
    result_section = "Chưa có run thực nghiệm."
    latest_section = "Chưa có dự báo mới nhất."
    if run_dir and (run_dir / "metrics_by_model_horizon.csv").exists():
        metrics = pd.read_csv(run_dir / "metrics_by_model_horizon.csv")
        result_section = markdown_table(metrics[["model", "horizon", "macro_f1", "balanced_accuracy", "brier", "log_loss", "recall_bear", "recall_stress"]], 60)
    if run_dir and (run_dir / "latest_outlook.json").exists():
        latest_section = "```json\n" + (run_dir / "latest_outlook.json").read_text(encoding="utf-8") + "\n```"
    readme = f"""# RAEMF-MC: Regime-Aware Explainable Multi-Horizon Forecasting with Monte Carlo

Tác giả và người phát triển mô hình: Nguyễn Hoài Nam.

RAEMF-MC dự báo trạng thái tương lai của VN-Index tại 20, 40 và 60 phiên theo bốn lớp `Bull`, `Sideway`, `Bear`, `Stress`. Repository này nhấn mạnh chống rò rỉ dữ liệu tương lai, đánh giá ngoài mẫu theo thời gian, xác suất có hiệu chỉnh và phân tích bất định.

```mermaid
flowchart LR
  A[data.csv] --> B[kiểm tra dữ liệu]
  B --> C[đặc trưng nhân quả]
  C --> D[Filtered HMM]
  C --> E[EGARCH Student-t]
  C --> F[RAEMF-MC EBM]
  D --> F
  E --> F
  F --> G[calibration]
  G --> H[metric, bootstrap, Monte Carlo, backtest exposure]
```

## Cài đặt

```bash
conda activate eda
python -m pip install -e .
```

## Chạy laptop mode

```bash
bash scripts/run_laptop.sh
```

## Quy tắc chống leakage

- Đặc trưng tại thời điểm t chỉ dùng dữ liệu đến t.
- Nhãn dùng `target_end_date_h`; train phải thỏa `target_end_date_h < validation_start`.
- Calibration chỉ fit trên validation, không fit trên test.
- Backtest dùng tín hiệu sau đóng cửa ngày t cho lợi suất từ t+1.

## Kết quả thực nghiệm mới nhất

{result_section}

## Dự báo mới nhất

{latest_section}

## Hình ảnh chính

Các hình được lưu trong `outputs/latest/figures/`, gồm VN-Index theo thời gian, phân phối lớp, so sánh metric, xác suất Filtered HMM, EGARCH và fan chart Monte Carlo.

## Giới hạn

Dữ liệu chỉ gồm VN-Index trong `data.csv`; không có dữ liệu vĩ mô, market breadth hoặc thành phần chỉ số. VN-Index không phải tài sản có thể giao dịch trực tiếp theo giả định đơn giản. Kết quả không phải lời khuyên đầu tư và không bảo đảm hiệu quả tương lai.
"""
    Path("README.md").write_text(readme, encoding="utf-8")
    methodology = """# Phương pháp

RAEMF-MC kết hợp đặc trưng kỹ thuật nhân quả, xác suất trạng thái hiện tại từ Filtered HMM, đặc trưng rủi ro EGARCH Student-t và EBM đa lớp để dự báo bốn trạng thái tại ba chân trời.
"""
    leakage = """# Phòng ngừa rò rỉ dữ liệu

Mọi split đều theo thời gian. Với từng horizon, quan sát train chỉ được dùng khi target end date nhỏ hơn boundary validation hoặc test. Không có centered rolling window, negative shift trong feature hoặc calibration fit trên test.
"""
    model_card = """# Model card

Mục tiêu là nghiên cứu định lượng VN-Index, không phải hệ thống khuyến nghị đầu tư. Confidence có thể là High, Medium, Low hoặc Uncertain.
"""
    reproducibility = """# Tái lập

Chạy `conda activate eda`, `python -m pip install -e .`, sau đó `bash scripts/run_laptop.sh`. Mỗi run lưu checksum dữ liệu, snapshot cấu hình, thông tin môi trường và git metadata.
"""
    technical = """# Báo cáo kỹ thuật

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
"""
    refs = """# Tài liệu tham khảo

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach.
- Efron, B. and Tibshirani, R. J. (1993). An introduction to the bootstrap.
- Bergmeir, C., Hyndman, R. J., and Benitez, J. M. (2016). Bagging exponential smoothing methods using STL decomposition and Box-Cox transformation.
- Niculescu-Mizil, A. and Caruana, R. (2005). Predicting good probabilities with supervised learning.
"""
    docs = {
        "methodology.md": methodology,
        "leakage_prevention.md": leakage,
        "model_card.md": model_card,
        "reproducibility.md": reproducibility,
        "technical_report.md": technical,
        "references.md": refs,
    }
    Path("docs").mkdir(exist_ok=True)
    for name, body in docs.items():
        Path("docs", name).write_text(body, encoding="utf-8")
