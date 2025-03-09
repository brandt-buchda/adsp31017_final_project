from sklearn.pipeline import Pipeline
from scripts.transformers import *


class PredictionPipeline(Pipeline):
    def __init__(self):
        steps = [
            ("cast transformer", SentimentTransformer(["cast"])),
            ("team transformer", SentimentTransformer(["director", "writer"], "team")),
            ("drop columns transformer", DropColumnsTransformer()),
        ]

        super(PredictionPipeline, self).__init__(steps)