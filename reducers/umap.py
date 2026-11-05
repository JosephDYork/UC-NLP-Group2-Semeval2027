from umap import UMAP

class UMAPReducer:
    def __init__(self, n_components=10, n_neighbors=15, min_dist=0.1, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.reducer = None

    def fit(self, vectors):
        self.reducer = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric="cosine",
            random_state=self.random_state,
        ).fit(vectors)

        return self

    def transform(self, vectors):
        if self.reducer is None:
            raise ValueError("Reducer must be fit before transformation")

        return self.reducer.transform(vectors)

    def fit_transform(self, vectors):
        return self.fit(vectors).transform(vectors)
