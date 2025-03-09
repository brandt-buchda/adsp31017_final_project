from sklearn.pipeline import Pipeline
from scripts.transformers import *


class PredictionPipeline(Pipeline):
    def __init__(self):
        steps = [
            ("cast transformer", SentimentTransformer(["cast"])),
            ("team transformer", SentimentTransformer(["director", "writers"], "team")),
            # ("plot emotion transformer", EmotionAnalysisTransformer()), TODO reenable
            ("genre transformer", GenreOneHotEncoder(min_occurrences=500)),
            ("drop columns transformer", DropColumnsTransformer()),
        ]

        super(PredictionPipeline, self).__init__(steps)