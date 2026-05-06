from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pathlib import Path
import argparse
from src.data.generator import generate_dataset
from src.data.io import save_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=int, default=5000)
    parser.add_argument('--output', type=str, default='data/synthetic/synthetic_trade_records.csv')
    args = parser.parse_args()
    df = generate_dataset(args.records)
    save_dataframe(df, args.output)
    print(f"Saved {len(df)} records to {args.output}")


if __name__ == '__main__':
    main()
