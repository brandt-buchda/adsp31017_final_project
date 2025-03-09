import itertools
import re

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from gensim.models import FastText
from transformers import pipeline

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

        for col in self.columns:
            X[col] = X[col].apply(
                lambda cell: re.split(r',\s*|\n', cell) if isinstance(cell, str) else cell)

        X[self.alias] = X[self.columns].apply(
            lambda row: list(itertools.chain.from_iterable(row)), axis=1)

        X[f'{self.alias}_names'] = X[self.alias].apply(
            lambda names: [clean_name(name) for name in names])

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


class EmotionAnalysisTransformer(TransformerMixin):
    def __init__(self,
                 model_name='bhadresh-savani/distilbert-base-uncased-emotion',
                 max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.emotion_analyzer = pipeline('text-classification',
                                         model=self.model_name,
                                         top_k=None)
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        emotion_data = []

        for row in X.itertuples(index=False):
            title, text = row.title, row.plot
            truncated_text = text[:self.max_length]

            emotions = self.emotion_analyzer(truncated_text)

            print(title, ": ", emotions)

            emotion_scores = {item['label']: item['score'] for item in emotions[0]}

            dominant_emotion = max(emotion_scores, key=emotion_scores.get)

            one_hot_dominant = {f"{label}_dominant_emotion": int(label == dominant_emotion) for label in self.emotion_labels}

            emotion_data.append({**emotion_scores, **one_hot_dominant})

        emotion_df = pd.DataFrame(emotion_data, index=X.index if isinstance(X, pd.Series) else None)

        return pd.concat([X.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)


class GenreOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, genres_column="genres", min_occurrences=500):
        self.mlb = MultiLabelBinarizer()
        self.genres_column = genres_column
        self.min_occurrences = min_occurrences

    def clean_genres(self, genre_list):
        cleaned_genres = []
        for genre in genre_list:
            split_genres = re.split(r'[,&/\- ]', genre.lower())
            cleaned_genres.extend([g.strip() for g in split_genres if g.strip()])

        return cleaned_genres

    def fit(self, X, y=None):
        all_genres = X[self.genres_column].apply(self.clean_genres)

        genre_counts = pd.Series([genre for sublist in all_genres for genre in
                                  sublist]).value_counts()

        self.keep_genres = genre_counts[
            genre_counts >= self.min_occurrences].index.tolist()

        filtered_genres = all_genres.apply(
            lambda x: [g for g in self.clean_genres(x) if
                       g in self.keep_genres])
        self.mlb.fit(filtered_genres)
        return self

    def transform(self, X):
        all_genres = X[self.genres_column].apply(self.clean_genres)
        genre_matrix = self.mlb.transform(all_genres)

        genre_df = pd.DataFrame(genre_matrix, columns=[f'genre_{g}' for g in self.mlb.classes_])

        return pd.concat([X, genre_df], axis=1)

class DropColumnsTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.drop(columns=['cast', 'writers', 'director', 'plot', 'genres'], inplace=True)

        return X