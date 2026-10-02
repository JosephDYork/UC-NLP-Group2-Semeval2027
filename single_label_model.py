class SingleLabelBaselineModel:
    def __init__(self, **kwargs):
        pass

    def fit(self, train):
        return self

    def predict(self, rows):
        return [f"{row.word}:0" for row in rows.itertuples()]
