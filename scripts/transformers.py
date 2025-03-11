import itertools
import re
import ast
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import pipeline

from config import *
from unidecode import unidecode


class ExternalDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.box_office_revenue = pd.read_csv(os.path.join(PROJECT_PATH, EXTERNAL_DATA_PATH, 'box_office_revenue.csv'), index_col=0)
        self.cpi_yearly = pd.read_csv(os.path.join(PROJECT_PATH, EXTERNAL_DATA_PATH, 'cpi_yearly.csv'), index_col=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X.merge(self.box_office_revenue, left_on="release_year", right_index=True, how="left")
        X = X.merge(self.cpi_yearly, left_on="release_year", right_index=True, how="left")
        return X


class BoxOfficeCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.box_office_revenue = pd.read_csv(os.path.join(PROJECT_PATH, EXTERNAL_DATA_PATH, 'box_office_revenue.csv'), index_col=0)
        self.latest_box_office_revenue = float(self.box_office_revenue.loc[2024].iloc[0])
        self.cpi_yearly = pd.read_csv(os.path.join(PROJECT_PATH, EXTERNAL_DATA_PATH, 'cpi_yearly.csv'), index_col=0)
        self.latest_cpi = float(self.cpi_yearly.loc[2024].iloc[0])
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X.dropna(subset=["box_office"])

        return X


class CollectionBoxOfficeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, csv_path="data/collection_box_office.csv"):
        self.csv_path = csv_path
        self.collection_avg_box_office = {}

    def fit(self, X, y=None):
        X = X.copy()
        X["collection"] = X["collection"].apply(self.safe_eval)
        X["collection_id"] = X["collection"].apply(lambda x: x["id"] if isinstance(x, dict) else None)
        collection_avg_box_office = X.groupby("collection_id")["box_office"].mean().dropna()

        collection_avg_box_office.to_csv(self.csv_path, header=["collection_box_office_average"])

        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["collection"] = X["collection"].apply(self.safe_eval)
        X["collection_id"] = X["collection"].apply(lambda x: x["id"] if isinstance(x, dict) else None)

        self.collection_avg_box_office = pd.read_csv(self.csv_path, index_col="collection_id")["collection_box_office_average"].to_dict()

        X["collection_box_office_average"] = X["collection_id"].map(self.collection_avg_box_office).fillna(0)
        X.drop(columns=["collection", "collection_id"], inplace=True)
        return X

    @staticmethod
    def safe_eval(value):
        if pd.isna(value):
            return None
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None
        try:
            parsed_value = ast.literal_eval(value)
            if isinstance(parsed_value, dict):
                return parsed_value
        except (ValueError, SyntaxError):
            return None
        return None


class RatingEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        rating_map = {
            np.nan: 0,
            "pg": 1,
            "tvpg": 2,
            "pg-13": 3,
            "nc-17": 4,
            "tvma": 5,
            "r": 6
        }
        X["rating"] = X["rating"].apply(lambda x: rating_map.get(x, 0))
        return X


class TopContributionsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categories, csv_path="data/top_contributions.csv"):
        """
        Parameters:
            categories (dict): A dictionary where keys are the category column names (e.g., 'director')
                               and values are the number (n) of top contributors to compute.
            csv_path (str): Path to the CSV file to save/load contributions.
        """
        self.categories = categories
        self.csv_path = csv_path

    @staticmethod
    def clean_name(name):
        """
        Lowercase the name and remove all non-alphabetical characters.
        """
        return re.sub(r'[^a-z]', '', name.lower())

    def fit(self, X, y=None):
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.csv_path)
        os.makedirs(output_dir, exist_ok=True)

        # This dictionary will map each category to its contributor average revenue
        self.contributions_ = {}
        # This list will collect rows to be saved into a single CSV file
        rows = []

        for category, n in self.categories.items():
            avg_dict = {}
            # Loop over each row to accumulate revenue and counts
            for _, row in X.iterrows():
                # Clean names: strip whitespace, clean, then filter empty strings
                members = [self.clean_name(name.strip())
                           for name in str(row[category]).split(",") if
                           name.strip()]
                for member in members:
                    if member not in avg_dict:
                        avg_dict[member] = {"total_revenue": 0, "count": 0}
                    avg_dict[member]["total_revenue"] += row["box_office"]
                    avg_dict[member]["count"] += 1

            # Compute the average revenue for each contributor
            avg_dict = {member: stats["total_revenue"] / stats["count"]
                        for member, stats in avg_dict.items() if
                        stats["count"] > 0}
            self.contributions_[category] = avg_dict

            # Append the computed values to the rows list
            for member, avg_rev in avg_dict.items():
                rows.append({
                    "category": category,
                    "member": member,
                    "avg_revenue": avg_rev
                })

        # Save all contributions to one CSV file
        df_contrib = pd.DataFrame(rows)
        df_contrib.to_csv(self.csv_path, index=False)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Load the precomputed contributions CSV file
        df_contrib = pd.read_csv(self.csv_path)

        # Create a dictionary mapping each category to its member:avg_revenue dictionary
        contributions_by_category = {}
        for category in self.categories:
            df_cat = df_contrib[df_contrib["category"] == category]
            contributions_by_category[category] = dict(
                zip(df_cat["member"], df_cat["avg_revenue"]))

        # For each category, compute the top contributions and add new columns
        for category, n in self.categories.items():
            avg_dict = contributions_by_category.get(category, {})

            def compute_contributions(names):
                # Clean and process the names in each row
                members = [self.clean_name(name.strip())
                           for name in str(names).split(",") if name.strip()]
                # Sort the members by average revenue (defaulting to -inf if missing)
                members = sorted(members,
                                 key=lambda x: avg_dict.get(x, float('-inf')),
                                 reverse=True)
                top_n = [avg_dict.get(member, float('nan')) for member in
                         members[:n]]
                # Pad the result with NaN if there are fewer than n contributors
                top_n += [float('nan')] * (n - len(top_n))
                return pd.Series(top_n)

            # Define new column names for the top contributors
            col_names = [f"{category}_contributor_{i + 1}" for i in range(n)]
            X[col_names] = X[category].apply(compute_contributions)

        return X

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
        self.expected_genres = ['action', 'adventure', 'comedy', 'crime', 'drama',
                           'family', 'mystery', 'romance', 'thriller']
        self.mlb = MultiLabelBinarizer(classes=self.expected_genres)
        self.genres_column = genres_column
        self.min_occurrences = min_occurrences

    def clean_genres(self, genre_list):
        cleaned_genres = []
        for genre in genre_list:
            split_genres = re.split(r'[,&/\- ]', genre.lower())
            cleaned_genres.extend([g.strip() for g in split_genres if g.strip()])

        return cleaned_genres

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        self.mlb.fit(self.expected_genres)
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
        X = X[(X["box_office"] >= 1e5) & (X["box_office"] <= 5e9)]
        X.drop(columns=['cast', 'writers', 'director', 'plot', 'genres', 'distributors'], inplace=True)
        X.dropna(subset=["box_office"], inplace=True)
        X["title_id"] = X["title"] + " (" + X["release_year"].astype(
            str) + ")"
        X.drop_duplicates(subset=["title_id"], inplace=True)
        X.set_index("title_id", inplace=True)
        X.drop(columns=["title"], inplace=True)
        return X