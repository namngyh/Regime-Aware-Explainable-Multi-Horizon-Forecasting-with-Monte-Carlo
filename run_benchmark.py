from pathlib import Path

from src.benchmark import run_benchmark


if __name__ == "__main__":
    run_benchmark(
        data_path=Path("data.csv"),
        output_dir=Path("outputs"),
        horizons=(5, 20, 60),
    )
