from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional

class BoxOfficeFeatures:
    def __init__(self,
                 title: str,
                 release_year: int,
                 box_office: float,
                 budget: float,
                 rating: str,
                 collection: str,
                 cast: list[str],
                 director: str,
                 writers: list[str],
                 distributors: list[str],
                 genres: list[str],
                 plot: str):
        self.title = title
        self.release_year = release_year
        self.box_office = box_office
        self.budget = budget
        self.rating = rating
        self.collection = collection
        self.cast = cast
        self.director = director
        self.writers = writers
        self.distributors = distributors
        self.genres = genres
        self.plot = plot

    def to_dataframe(self):
        pass