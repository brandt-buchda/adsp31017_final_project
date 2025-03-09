import pandas as pd
from typing import Union

class DataLoader:
    def __init__(self, schema: dict=None, schema_mapping: dict=None):
        self.schema = schema
        self.schema_mapping = schema_mapping

    def load_csv(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)

            if self.schema is not None:
                df.rename(columns=self.schema_mapping, inplace=True)
            return self.validate_data(df)

        except Exception as e:
            raise ValueError(f"Error loading from CSV: {e}")

    def load_manual(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])
        return self.validate_data(df)

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate column names, enforce data types, and drop extra columns.
        """
        if self.schema:
            missing_cols = [col for col in self.schema.keys() if
                            col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing expected columns: {missing_cols}")

            df = df[list(self.schema.keys())]

            for col, dtype in self.schema.items():
                if dtype == list[str]:
                    df[col] = df[col].apply(
                        lambda x: x.split(", ") if isinstance(x, str) else [])
                else:
                    df[col] = df[col].astype(dtype)

        return df
