from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
from pipelines.training_pipeline import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/synthetic/synthetic_trade_records.csv')
    args = parser.parse_args()
    bundle = run_training(args.input)
    print(f"Training complete. Train rows={bundle['train_rows']}, test rows={bundle['test_rows']}")


if __name__ == '__main__':
    main()
