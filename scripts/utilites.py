import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def column_stats(df, metrics=None):
    metric_functions = {
        "Data Type": lambda: df.dtypes,
        "Count": lambda: df.count(),
        "Missing Values": lambda: df.isna().sum(),
        "Missing Value Rate": lambda: df.isna().sum() / df.shape[0] * 100,
        "Unique Values": lambda: df.nunique(),
        "Unique Rate": lambda: df.nunique() / df.shape[0] * 100,
        "Mean": lambda: df.select_dtypes(include=["number"]).mean(),
        "Median": lambda: df.select_dtypes(include=["number"]).median(),
        "Mode": lambda: df.mode().iloc[0],
        "Min": lambda: df.select_dtypes(include=["number"]).min(),
        "Max": lambda: df.select_dtypes(include=["number"]).max(),
    }

    if metrics is None:
        metrics = list(metric_functions.keys())  # Convert keys to a list

    stats = pd.DataFrame()

    for metric in metrics:
        if metric in metric_functions:
            stats[metric] = metric_functions[metric]()

    return stats


class DataAdjuster:
    def __init__(self, adjust_cpi=True, adjust_gross=True, apply_boxcox=True,
                 boxcox_lambda=None, cpi_data_path=None, gross_data_path=None):
        self.adjust_cpi = adjust_cpi
        self.adjust_gross = adjust_gross
        self.apply_boxcox = apply_boxcox
        self.boxcox_lambda = boxcox_lambda

        # Read and store CPI and gross data
        self.cpi_data = pd.read_csv(cpi_data_path, index_col=0) if cpi_data_path else None
        self.gross_data = pd.read_csv(gross_data_path, index_col=0) if gross_data_path else None
        self.date_ref_col = None

        if self.cpi_data is not None:
            self.cpi_data.index = self.cpi_data.index.astype(int)
            self.cpi_col = self.cpi_data.columns[0]  # store the CPI column name
        if self.gross_data is not None:
            self.gross_data.index = self.gross_data.index.astype(int)
            self.gross_col = self.gross_data.columns[0]  # store the gross column name

    def capture_date_ref(self, df, date_ref_col):
        self.date_ref_col = date_ref_col

    def get_latest_values(self):
        latest_cpi = self.cpi_data.iloc[-1, 0] if self.adjust_cpi and self.cpi_data is not None else None
        latest_gross = self.gross_data.iloc[-1, 0] if self.adjust_gross and self.gross_data is not None else None
        return latest_cpi, latest_gross

    def apply_adjustments(self, df, target, date_ref_col, *cols):
        cols = (target,) + cols
        self.capture_date_ref(df, date_ref_col)
        latest_cpi, latest_gross = self.get_latest_values()

        if self.adjust_cpi and self.cpi_data is not None:
            df = df.merge(self.cpi_data[[self.cpi_col]], left_on=date_ref_col, right_index=True, how="left")
        if self.adjust_gross and self.gross_data is not None:
            df = df.merge(self.gross_data[[self.gross_col]], left_on=date_ref_col, right_index=True, how="left")

        for col in cols:
            if self.adjust_cpi and self.cpi_data is not None:
                df[col] *= latest_cpi / df[self.cpi_col]
            if self.adjust_gross and self.gross_data is not None:
                df[col] *= latest_gross / df[self.gross_col]
            if self.apply_boxcox and col == target:
                if self.boxcox_lambda is None:
                    df[col], self.boxcox_lambda = boxcox(df[col])
                else:
                    df[col], _ = boxcox(df[col], lmbda=self.boxcox_lambda)

        # Drop the CPI and gross columns by name instead of by position
        cols_to_drop = []
        if self.adjust_cpi and self.cpi_data is not None:
            cols_to_drop.append(self.cpi_col)
        if self.adjust_gross and self.gross_data is not None:
            cols_to_drop.append(self.gross_col)
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        return df

    def reverse_adjustments(self, df, target, date_ref_col, *cols):
        cols = (target,) + cols
        latest_cpi, latest_gross = self.get_latest_values()

        if self.adjust_cpi and self.cpi_data is not None:
            df = df.merge(self.cpi_data[[self.cpi_col]], left_on=date_ref_col, right_index=True, how="left")
        if self.adjust_gross and self.gross_data is not None:
            df = df.merge(self.gross_data[[self.gross_col]], left_on=date_ref_col, right_index=True, how="left")

        for col in cols:
            if self.adjust_cpi and self.cpi_data is not None:
                df[col] *= df[self.cpi_col] / latest_cpi  # Reverse CPI adjustment
            if self.adjust_gross and self.gross_data is not None:
                df[col] *= df[self.gross_col] / latest_gross  # Reverse gross adjustment
            if self.apply_boxcox and col == target:
                df[col] = inv_boxcox(df[col], self.boxcox_lambda)

        # Drop the CPI and gross columns by name, preserving the date column
        cols_to_drop = []
        if self.adjust_cpi and self.cpi_data is not None:
            cols_to_drop.append(self.cpi_col)
        if self.adjust_gross and self.gross_data is not None:
            cols_to_drop.append(self.gross_col)
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        return df


