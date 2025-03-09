import re

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from config import *
from unidecode import unidecode


class SentimentTransformer(TransformerMixin):
    def __init__(self, columns, alias=None):

        self.columns = columns
        self.alias = alias if alias else columns[0]
        self._sentiments = pd.read_csv(os.path.join(PROJECT_PATH, DATA_PATH, 'sentiment.csv'), index_col=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        def clean_name(name):
            name = name.strip()
            name = unidecode(name)
            name = re.sub(r'[^a-zA-Z\s]', '', name)
            name = " ".join(name.split())
            return name.lower()

        X[self.alias] = X[self.columns].apply(
            lambda row: ", ".join(row.fillna("")), axis=1)

        X[self.alias] = X[self.alias].apply(
            lambda names: ", ".join(
                [clean_name(name) for name in names.split(",")]))

        X[f'{self.alias}_names'] = X[self.alias].apply(
            lambda names: [clean_name(name) for name in names.split(",")])

        sentiment_columns = self._sentiments.select_dtypes(
            include='number').columns.tolist()

        for i, row in X.iterrows():
            names = row[f'{self.alias}_names']

            valid_names = [name for name in names if name in self._sentiments.index]

            if valid_names:
                relevant_sentiments = self._sentiments.loc[valid_names]

                for col in sentiment_columns:
                    if not relevant_sentiments[col].isnull().all():
                        X.at[i, f'{self.alias}_{col}'] = relevant_sentiments[
                            col].mean()
                    else:
                        X.at[i, f'{self.alias}_{col}'] = np.nan
            else:
                for col in sentiment_columns:
                    X.at[i, f'{self.alias}_{col}'] = np.nan

        X.drop(columns=[f'{self.alias}_names'], inplace=True)
        if self.alias != self.columns[0]:
            X.drop(columns=[f'{self.alias}'], inplace=True)

        return X

class DropColumnsTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.drop(columns=['cast', 'writer', 'director'], inplace=True)

        return X