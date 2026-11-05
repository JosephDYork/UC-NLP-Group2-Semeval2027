import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(
        self, input_dim, latent_dim, hidden_dims=(500, 500, 2000), activation=nn.GELU
    ):
        super().__init__()
        encoder_layers, decoder_layers = [], []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend((nn.Linear(previous_dim, hidden_dim), activation()))
            previous_dim = hidden_dim
        encoder_layers.append(nn.Linear(previous_dim, latent_dim))

        previous_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend((nn.Linear(previous_dim, hidden_dim), activation()))
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


class AutoencoderReducer:
    def __init__(
        self,
        n_components=20,
        epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=0.0,
        activation="gelu",
        standardize=True,
        hidden_dims=(500, 500, 2000),
        random_state=42,
    ):
        self.n_components = n_components
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.activation = activation
        self.standardize = standardize
        self.hidden_dims = hidden_dims
        self.random_state = random_state
        self.device = "cuda"
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, vectors):
        torch.manual_seed(self.random_state)
        scaled_vectors = (
            self.scaler.fit_transform(vectors) if self.standardize else vectors
        )

        inputs = torch.from_numpy(scaled_vectors.astype(np.float32))
        activations = {"gelu": nn.GELU, "relu": nn.ReLU}
        self.model = Autoencoder(
            inputs.shape[1],
            self.n_components,
            hidden_dims=self.hidden_dims,
            activation=activations[self.activation],
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loss_function = nn.MSELoss()

        for _ in range(self.epochs):
            for (batch,) in DataLoader(
                TensorDataset(inputs), batch_size=self.batch_size, shuffle=True
            ):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = loss_function(self.model(batch), batch)
                loss.backward()
                optimizer.step()

        return self

    def transform(self, vectors):
        if self.model is None:
            raise ValueError("Reducer must be fit before transformation")

        transformed_vectors = (
            self.scaler.transform(vectors) if self.standardize else vectors
        )

        inputs = torch.from_numpy(transformed_vectors.astype(np.float32)).to(
            self.device
        )

        with torch.inference_mode():
            return self.model.encoder(inputs).cpu().numpy()

    def fit_transform(self, vectors):
        return self.fit(vectors).transform(vectors)
