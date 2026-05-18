"""Lightweight JAX K-means for COT Policy observation clustering.

Usage:
    centroids = fit_kmeans(embeddings, n_clusters=64, n_iters=100)
    cluster_ids = assign_clusters(embeddings, centroids)
"""

import jax
import jax.numpy as jnp
from jax import Array


def fit_kmeans(data: Array, n_clusters: int, n_iters: int = 100, seed: int = 0) -> Array:
    """Fit K-means on data using Lloyd's algorithm.

    Args:
        data: shape (N, dim)
        n_clusters: number of clusters K
        n_iters: number of Lloyd iterations
        seed: random seed for centroid initialization

    Returns:
        centroids: shape (n_clusters, dim)
    """
    rng = jax.random.PRNGKey(seed)
    indices = jax.random.choice(rng, data.shape[0], shape=(n_clusters,), replace=False)
    centroids = data[indices]

    def step(centroids, _):
        dists = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)  # (N, K)
        assignments = jnp.argmin(dists, axis=-1)                                    # (N,)
        new_centroids = jax.vmap(
            lambda k: jnp.where(
                (assignments == k).any(),
                data[assignments == k].mean(axis=0),
                centroids[k],
            )
        )(jnp.arange(n_clusters))
        return new_centroids, None

    centroids, _ = jax.lax.scan(step, centroids, None, length=n_iters)
    return centroids


def assign_clusters(data: Array, centroids: Array) -> Array:
    """Assign each data point to its nearest centroid.

    Args:
        data: shape (N, dim)
        centroids: shape (n_clusters, dim)

    Returns:
        cluster_ids: shape (N,) integer cluster assignments
    """
    dists = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)  # (N, K)
    return jnp.argmin(dists, axis=-1)                                            # (N,)
