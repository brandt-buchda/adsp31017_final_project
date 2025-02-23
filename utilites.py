import pandas as pd
import numpy as np


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