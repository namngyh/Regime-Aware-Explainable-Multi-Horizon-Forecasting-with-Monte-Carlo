from __future__ import annotations
import argparse
from src.data import load_vnindex_csv
from src.reporting.benchmark_report import generate_core_figures
def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",required=True); p.add_argument("--data",default="data.csv"); a=p.parse_args(); generate_core_figures(a.run_dir,load_vnindex_csv(a.data))
if __name__=="__main__": main()

