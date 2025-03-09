from sklearn.pipeline import Pipeline
from scripts.transformers import *


class PredictionPipeline(Pipeline):
    def __init__(self):
        steps = [
            ("external data transformer", ExternalDataTransformer()),
            ("box office cleaner transformer", BoxOfficeCleanerTransformer()),
            ("collection transformer", CollectionBoxOfficeTransformer()),
            ("rating transformer", RatingEncoder()),
            ("contribution transformer", TopContributionsTransformer(categories={
                "cast": 5,
                "director": 1,
                "writers": 3,
                "distributors": 1})),
            ("cast transformer", SentimentTransformer(["cast"])),
            ("team transformer", SentimentTransformer(["director", "writers"], "team")),
            # ("plot emotion transformer", EmotionAnalysisTransformer()),
            ("genre transformer", GenreOneHotEncoder(min_occurrences=500)),
            ("drop columns transformer", DropColumnsTransformer()),
        ]

        super(PredictionPipeline, self).__init__(steps)