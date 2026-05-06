#!/usr/bin/env bash
set -e
python scripts/generate_sample_data.py --records 5000
python scripts/train_models.py --input data/synthetic/synthetic_trade_records.csv
streamlit run app/streamlit_app.py
