import argparse
import pandas as pd
from config import *
from scripts.data_loader import DataLoader
from scripts.pipeline import PredictionPipeline
from scripts.utilites import column_stats

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def parse_args():
    parser = argparse.ArgumentParser(description="Load data from CSV or manual input.")
    parser.add_argument("--csv", type=str, help="Path to CSV file containing data.")
    parser.add_argument("--test", action="store_true", help="Flag to run in test mode (only transform and predict).")
    return parser.parse_args()

def main():
    args = parse_args()
    loader = DataLoader(schema_mapping=DEFAULT_SCHMA_MAPPING)
    pipeline = PredictionPipeline()

    if args.csv:
        df = loader.load_csv(args.csv)
    else:
        print("No CSV provided. Please enter manual input:")
        manual_input = input(f"Enter values for {', '.join(DEFAULT_SCHEMA.keys())}: ").split()

        if len(manual_input) != len(DEFAULT_SCHEMA):
            print(f"Error: Expected {len(DEFAULT_SCHEMA)} values, but got {len(manual_input)}.")
            print(f"Expected order: {list(DEFAULT_SCHEMA.keys())}")
            return

        data_dict = {col: DEFAULT_SCHEMA[col](value) for col, value in zip(DEFAULT_SCHEMA.keys(), manual_input)}
        df = loader.load_manual(data_dict)

    if args.test:
        print("Running in test mode: Transforming data and making predictions...")
        transformed = pipeline.transform(df)
        transformed.set_index("title", inplace=True)
        transformed.to_csv("test.csv", index=True)
    else:
        print("Training mode: Fitting and transforming pipeline...")
        transformed = pipeline.fit_transform(df)
        transformed.set_index("title", inplace=True)
        transformed.to_csv("train.csv", index=True)

    # print("Making predictions...")
    # predictions = pipeline.predict(transformed)
    # print(predictions)

if __name__ == "__main__":
    main()
