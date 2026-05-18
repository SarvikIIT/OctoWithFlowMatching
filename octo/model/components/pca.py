"""Lightweight JAX PCA for COT Policy observation embedding compression.

Usage:
    mean, components = fit_pca(embeddings, n_components=32)
    reduced = transform_pca(embeddings, mean, components)
"""

import jax.numpy as jnp
from jax import Array


def fit_pca(data: Array, n_components: int) -> tuple[Array, Array]:
    """Compute PCA components from data using SVD.

    Args:
        data: shape (N, emb_dim) — observation embeddings, one per sample
        n_components: number of principal components to keep

    Returns:
        mean: shape (emb_dim,) — per-feature mean, used to center data
        components: shape (n_components, emb_dim) — top-k eigenvectors (rows)
    """
    mean = data.mean(axis=0)                         # (emb_dim,)
    centered = data - mean                           # (N, emb_dim)
    _, _, Vt = jnp.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]                   # (n_components, emb_dim)
    return mean, components


def transform_pca(data: Array, mean: Array, components: Array) -> Array:
    """Project data onto pre-computed PCA components.

    Args:
        data: shape (N, emb_dim)
        mean: shape (emb_dim,) — from fit_pca
        components: shape (n_components, emb_dim) — from fit_pca

    Returns:
        reduced: shape (N, n_components)
    """
    centered = data - mean                           # (N, emb_dim)
    return centered @ components.T                   # (N, n_components)
