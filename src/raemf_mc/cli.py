"""Command-line interface for RAEMF-MC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raemf_mc.config import load_config
from raemf_mc.data.validation import validate_data_file
from raemf_mc.ops.ingest import IngestError, ingest_latest
from raemf_mc.pipeline import run_pipeline
from raemf_mc.reporting.current_monitor import generate_current_monitor
from raemf_mc.reporting.plots import generate_all_plots
from raemf_mc.reporting.report_builder import build_docs_and_readme, build_run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raemf-mc")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate-data")
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", default="outputs/data_validation")

    p = sub.add_parser("run")
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)

    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)

    p = sub.add_parser("plots")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data", default="data.csv")

    p = sub.add_parser("forecast-latest")
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)

    p = sub.add_parser("current-report")
    p.add_argument("--data", default="VNINDEX_Daily.csv")
    p.add_argument("--baseline-run", default="outputs/latest")
    p.add_argument("--config", default="configs/laptop.yaml")
    p.add_argument("--output-dir", default="outputs/current_monitor")
    p.add_argument("--readme", default="README.md")

    p = sub.add_parser("reproduce")
    p.add_argument("--data", default="data.csv")
    p.add_argument("--config", default="configs/laptop.yaml")

    p = sub.add_parser("ingest-data", help="Nạp file DataPro mới nhất từ incoming/ vào lịch sử chính")
    p.add_argument("--data", default="VNINDEX_Daily.csv")
    p.add_argument("--incoming-dir", default="incoming")
    p.add_argument("--backup-dir", default="backups")

    p = sub.add_parser("daily", help="Chu trình hằng ngày: ingest -> validate -> current-report")
    p.add_argument("--data", default="VNINDEX_Daily.csv")
    p.add_argument("--incoming-dir", default="incoming")
    p.add_argument("--backup-dir", default="backups")
    p.add_argument("--baseline-run", default="outputs/latest")
    p.add_argument("--config", default="configs/laptop.yaml")
    p.add_argument("--output-dir", default="outputs/current_monitor")
    p.add_argument("--readme", default="README.md")
    return parser


def _run_ingest(args: argparse.Namespace) -> dict[str, object]:
    result = ingest_latest(target_csv=args.data, incoming_dir=args.incoming_dir, backup_dir=args.backup_dir)
    summary = result.to_dict()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "validate-data":
        profile = validate_data_file(args.data, args.output_dir)
        print(json.dumps(profile, indent=2, ensure_ascii=False))
    elif args.cmd in {"run", "forecast-latest"}:
        config = load_config(args.config)
        run_dir = run_pipeline(args.data, config)
        print(run_dir)
    elif args.cmd == "report":
        build_run_report(args.run_dir)
        build_docs_and_readme(args.run_dir)
        print(Path(args.run_dir) / "report.md")
    elif args.cmd == "current-report":
        config = load_config(args.config)
        output_dir = generate_current_monitor(args.data, args.baseline_run, config, args.output_dir, args.readme)
        print(output_dir / "report_for_nonspecialists.md")
    elif args.cmd == "plots":
        figures = generate_all_plots(args.run_dir, args.data)
        print(f"Generated {len(figures)} figures in {Path(args.run_dir) / 'figures'}")
    elif args.cmd == "reproduce":
        config = load_config(args.config)
        run_dir = run_pipeline(args.data, config)
        print(run_dir)
    elif args.cmd == "ingest-data":
        try:
            _run_ingest(args)
        except IngestError as exc:
            raise SystemExit(f"INGEST TỪ CHỐI: {exc}") from exc
    elif args.cmd == "daily":
        try:
            summary = _run_ingest(args)
        except IngestError as exc:
            raise SystemExit(f"INGEST TỪ CHỐI: {exc}") from exc
        if summary["status"] == "no_new_file":
            print(f"Không có file mới trong {args.incoming_dir}/ — chạy với dữ liệu hiện có.")
        validate_data_file(args.data, "outputs/data_validation")
        config = load_config(args.config)
        output_dir = generate_current_monitor(args.data, args.baseline_run, config, args.output_dir, args.readme)
        print(output_dir / "report_for_nonspecialists.md")


if __name__ == "__main__":
    main()
