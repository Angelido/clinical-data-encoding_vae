from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import TransformerMixin
from skorch import NeuralNet


class _MIEOVAEModule(nn.Module):
    """
    Variational autoencoder for mixed continuous/binary clinical features.
    Expects the input to optionally contain a trailing mask (same length as the
    feature vector) that flags observed entries (1 = observed, 0 = missing).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        data_dim: int = 100,
        mask_dim: Optional[int] = None,
        hidden_dims: Optional[list[int]] = None,
        binary: Optional[int] = None,
        mask_percentage: float = 0.0,
    ):
        """
        Args:
            latent_dim: latent space dimensionality.
            data_dim: number of clinical features to reconstruct (without mask).
            mask_dim: dimension of the appended mask; defaults to data_dim when None.
            hidden_dims: encoder hidden sizes; decoder is symmetric.
            binary: number of binary features at the end of the data vector.
            mask_percentage: probability of masking each feature during training.
        """
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.binary_features = binary or 0
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.mask_dim = data_dim if mask_dim is None else mask_dim
        self.mask_percentage = mask_percentage
        self.input_dim = self.data_dim + self.mask_dim

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=hidden_dims[0]),
            nn.Tanh(),
        )
        for i in range(len(hidden_dims) - 1):
            self.encoder.append(
                nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i + 1])
            )
            self.encoder.append(nn.Tanh())

        # --- mu - logvar ---
        self.mu = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)
        self.logvar = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dims[-1]),
            nn.Tanh(),
        )
        for i in range(len(hidden_dims) - 1):
            self.decoder.append(
                nn.Linear(
                    in_features=hidden_dims[-i - 1], out_features=hidden_dims[-i - 2]
                )
            )
            self.decoder.append(nn.Tanh())

        self.binary_dec = nn.Sequential(
            nn.Linear(in_features=hidden_dims[0], out_features=self.binary_features),
            nn.Sigmoid(),
        )
        self.continuous_dec = nn.Linear(
            in_features=hidden_dims[0],
            out_features=self.data_dim - self.binary_features,
        )

    def split_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (data, mask) from concatenated input."""
        if self.mask_dim > 0:
            x_data = x[:, : self.data_dim]
            null_mask = x[:, self.data_dim : self.data_dim + self.mask_dim]
        else:
            x_data = x
            null_mask = None
        return x_data, null_mask

    def _apply_random_mask(self, x_data: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_percentage <= 0:
            return x_data
        keep_mask = torch.rand_like(x_data) > self.mask_percentage
        x_masked = x_data * keep_mask.float()
        # Mark artificially masked entries as -1 to mimic missing values.
        return x_masked - (~keep_mask).float()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, encoder_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(encoder_input)
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        return mu, logvar

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        reconstructed = self.decoder(z)
        binary_rec = self.binary_dec(reconstructed)
        continuous_rec = self.continuous_dec(reconstructed)
        return torch.cat((continuous_rec, binary_rec), dim=1)

    def encode_decode(
        self, x: torch.Tensor, apply_random_mask: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_data, null_mask = self.split_input(x)
        x_data_masked = (
            self._apply_random_mask(x_data) if apply_random_mask else x_data
        )
        encoder_input = (
            torch.cat((x_data_masked, null_mask), dim=1)
            if null_mask is not None
            else x_data_masked
        )
        mu, logvar = self.encode(encoder_input)
        z = self.reparameterize(mu, logvar)
        reconstructed = self._decode(z)
        return reconstructed, mu, logvar

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode_decode(x, apply_random_mask=self.training)

    def encode_only(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic mean embeddings (mu) without random masking."""
        mu, _ = self.encode(self._prepare_encoder_input_for_eval(x))
        return mu

    def _prepare_encoder_input_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        x_data, null_mask = self.split_input(x)
        if null_mask is not None:
            return torch.cat((x_data, null_mask), dim=1)
        return x_data

    def kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL divergence between N(mu, var) and N(0, 1)
        return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()

    def loss_from_output(
        self,
        y_pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        x_full: torch.Tensor,
        beta: float = 1.0,
        binary_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstructed, mu, logvar = y_pred
        x_data, null_mask = self.split_input(x_full)
        observed = null_mask if null_mask is not None else torch.ones_like(x_data)
        cont_dim = self.data_dim - self.binary_features

        mse_loss = F.mse_loss(
            reconstructed[:, :cont_dim] * observed[:, :cont_dim],
            x_data[:, :cont_dim] * observed[:, :cont_dim],
            reduction="mean",
        )
        bce_loss = F.binary_cross_entropy(
            reconstructed[:, cont_dim:] * observed[:, cont_dim:],
            x_data[:, cont_dim:] * observed[:, cont_dim:],
            reduction="mean",
        )
        reconstruction_loss = mse_loss + binary_weight * bce_loss
        kl_loss = self.kl(mu, logvar)
        total_loss = reconstruction_loss + beta * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def saveModel(self, path: str):
        """Save the model to a file"""
        torch.save(self, path)

    def freeze(self):
        """Freeze the model parameters"""
        for param in self.parameters():
            param.requires_grad = False


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class MIEOVAE(NeuralNet, TransformerMixin):
    """
    Skorch + sklearn compatible wrapper around the underlying VAE torch module.

    - fit(X, y=None, X_unlabeled=...) trains unsupervised (reconstruction) on X (+ optional unlabeled).
    - transform(X) returns deterministic embeddings (mu).

    Expected input layout:
        X = [values_filled | null_mask]
    where null_mask has the same length as values_filled and uses 1 for observed entries, 0 for missing.
    """

    def __init__(
        self,
        module=_MIEOVAEModule,
        beta: float = 1.0,
        binary_weight: float = 1.0,
        **kwargs,
    ):
        self.beta = beta
        self.binary_weight = binary_weight
        super().__init__(module=module, **kwargs)

    def fit(self, X, y=None, X_unlabeled=None, **fit_params):  # noqa: N802
        X_labeled = _to_numpy(X)
        X_extra = _to_numpy(X_unlabeled) if X_unlabeled is not None else None
        X_train = np.concatenate([X_labeled, X_extra], axis=0) if X_extra is not None else X_labeled
        return super().fit(X_train, X_train, **fit_params)

    def get_loss(self, y_pred, y_true, X=None, training=False):  # noqa: N802
        x_tensor = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        total_loss, _, _ = self.module_.loss_from_output(
            y_pred=y_pred, x_full=x_tensor, beta=self.beta, binary_weight=self.binary_weight
        )
        return total_loss

    def transform(self, X):
        X_np = _to_numpy(X)
        X_tensor = torch.as_tensor(X_np, device=self.device, dtype=torch.float32)
        self.module_.eval()
        with torch.no_grad():
            mu = self.module_.encode_only(X_tensor)
        return mu.detach().cpu().numpy()


if __name__ == "__main__":
    module = _MIEOVAEModule(data_dim=60, mask_dim=60, hidden_dims=[60, 40, 30, 20, 3])
    print(module)
