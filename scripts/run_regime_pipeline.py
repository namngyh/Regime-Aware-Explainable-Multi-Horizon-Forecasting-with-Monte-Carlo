from __future__ import annotations
import argparse, logging
from src.pipeline import run_pipeline

def main() -> None:
    p=argparse.ArgumentParser(description="Run RAEMF-MC")
    p.add_argument("--data",default="data.csv"); p.add_argument("--config",default="config/model_config.yaml"); p.add_argument("--output",default="outputs/runs"); p.add_argument("--quick-mode",action="store_true"); args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s"); print(run_pipeline(args.data,args.config,args.output,args.quick_mode))
if __name__=="__main__": main()

