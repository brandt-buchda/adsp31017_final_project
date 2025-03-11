import pandas as pd
import re
from config import *

class DataLoader:
    def __init__(self, schema: dict=DEFAULT_SCHEMA, schema_mapping: dict=None):
        self.schema = schema
        self.schema_mapping = schema_mapping

    def load_csv(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)

            if self.schema_mapping is not None:
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

            # Ensure we're working with a copy and avoid SettingWithCopyWarning
            df = df.loc[:, list(self.schema.keys())]

            for col, dtype in self.schema.items():
                if dtype == list[str]:
                    df.loc[:, col] = df[col].apply(
                        lambda x: re.split(r",\s*|\n", x) if isinstance(x, str) else [])
                else:
                    df.loc[:, col] = df[col].astype(dtype)

        return df
