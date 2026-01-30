import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
import pymc as pm
from pymc import math as pm_math
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("ppcx")

RANDOM_SEED = 8927
EPS = 1e-12
rng = np.random.default_rng(RANDOM_SEED)


""" Mixture models with spatial priors"""


def build_marginalized_mixture_model_gpu(
    data: jnp.ndarray,
    prior_probs: jnp.ndarray,
    sectors: dict[str, Any],
    mu_params: dict[str, Any] | None = None,
    sigma_params: dict[str, Any] | None = None,
):
    """
    Build a marginalized mixture PyMC model (no discrete z), GPU/JAX version.
    Returns a PyMC model object (not sampled).
    """
    n_data = data.shape[0]
    n_features = data.shape[1]
    k = len(sectors)

    model = pm.Model(
        coords={"obs": range(n_data), "cluster": range(k), "feature": range(n_features)}
    )

    # Parse mu and sigma parameters
    if mu_params is None:
        mu_params = {"mu": 0, "sigma": 1}
    if sigma_params is None:
        sigma_params = {"sigma": 1}

    with model:
        obs_data = pm.Data("obs_data", data, dims=("obs", "feature"))
        prior_w = pm.Data(
            "prior_w", prior_probs.reshape(n_data, k), dims=("obs", "cluster")
        )

        mu = pm.Normal(
            "mu", mu_params["mu"], mu_params["sigma"], dims=("cluster", "feature")
        )
        sigma = pm.HalfNormal(
            "sigma", sigma_params["sigma"], dims=("cluster", "feature")
        )

        # Log weights with small constant to avoid log(0)
        log_w = pm.Deterministic(
            "log_w", pm_math.log(prior_w + 1e-12), dims=("obs", "cluster")
        )

        # Per-cluster log-likelihood
        x_centered = (obs_data[:, None, :] - mu[None, :, :]) / sigma[None, :, :]
        logp_feat = -0.5 * (
            pm_math.log(2 * jnp.pi) + 2 * pm_math.log(sigma[None, :, :]) + x_centered**2
        )
        logp_clusters = logp_feat.sum(axis=2)  # (obs, cluster)

        # Mixture log likelihood (marginalized over clusters)
        log_mix = pm.math.logsumexp(logp_clusters + log_w, axis=1)  # (obs,)

        # Total logp as Potential
        pm.Potential("mixture_logp", log_mix.sum())

    logger.info("Marginalized mixture model (un-sampled, GPU/JAX) created.")
    return model


def build_marginalized_mixture_model(
    data: np.ndarray,
    prior_probs: np.ndarray,
    sectors: dict[str, Any],
    mu_params: dict[str, Any] | None = None,
    sigma_params: dict[str, Any] | None = None,
):
    """
    Build a marginalized mixture PyMC model (no discrete z).
    Returns a PyMC model object (not sampled).
    """
    n_data = data.shape[0]
    n_features = data.shape[1]
    k = len(sectors)

    model = pm.Model(
        coords={"obs": range(n_data), "cluster": range(k), "feature": range(n_features)}
    )

    # Parse mu and sigma parameters
    if mu_params is None:
        mu_params = {"mu": 0, "sigma": 1}
    if sigma_params is None:
        sigma_params = {"sigma": 1}

    with model:
        obs_data = pm.Data("obs_data", data, dims=("obs", "feature"))
        prior_w = pm.Data(
            "prior_w", prior_probs.reshape(n_data, k), dims=("obs", "cluster")
        )

        mu = pm.Normal(
            "mu", mu_params["mu"], mu_params["sigma"], dims=("cluster", "feature")
        )
        sigma = pm.HalfNormal(
            "sigma", sigma_params["sigma"], dims=("cluster", "feature")
        )

        # Log weights with small constant to avoid log(0)
        log_w = pm.Deterministic(
            "log_w", pm_math.log(prior_w + 1e-12), dims=("obs", "cluster")
        )

        # Per-cluster log-likelihood
        x_centered = (obs_data[:, None, :] - mu[None, :, :]) / sigma[None, :, :]
        logp_feat = -0.5 * (
            pm_math.log(2 * np.pi) + 2 * pm_math.log(sigma[None, :, :]) + x_centered**2
        )
        logp_clusters = logp_feat.sum(axis=2)  # (obs, cluster)

        # Mixture log likelihood (marginalized over clusters)
        log_mix = pm.logsumexp(logp_clusters + log_w, axis=1)  # (obs,)

        # Total logp as Potential
        pm.Potential("mixture_logp", log_mix.sum())

    logger.info("Marginalized mixture model (un-sampled) created.")
    return model


def build_discrete_marginalized_mixture_model(
    data: np.ndarray,
    prior_probs: np.ndarray,
    sectors: dict[str, Any],
) -> pm.Model:
    """
    Simple marginalized mixture model with discrete cluster assignments.
    The discrete assigments require Metropolis sampling (not NUTS) that is not efficient, but it can be used for testing. Prefer the marginalized model if possible.
    """

    n_features = data.shape[1]
    n_data = data.shape[0]
    k = len(sectors)
    model = pm.Model(
        coords={"obs": range(n_data), "cluster": range(k), "feature": range(n_features)}
    )
    with model:
        # Cluster means
        mu = pm.Normal("mu", mu=0, sigma=1, dims=("cluster", "feature"))

        # Cluster standard deviations (diagonal covariance)
        sigma = pm.HalfNormal("sigma", sigma=1, dims=("cluster", "feature"))

        # Cluster assignments with spatial priors
        z = pm.Categorical("z", p=prior_probs, dims="obs")

        # Likelihood: each point comes from its assigned cluster
        observations = pm.Normal(
            "x_obs", mu=mu[z], sigma=sigma[z], observed=data, dims=("obs", "feature")
        )

        # Sample from the prior predictive distribution
        # prior_samples = pm.sample_prior_predictive(100)
        # fig, ax = plt.subplots(figsize=(8, 4))
        # az.plot_dist(
        #     data,
        #     kind="hist",
        #     color="C1",
        #     hist_kwargs={"alpha": 0.6},
        #     label="observed",
        # )
        # az.plot_dist(
        #     prior_samples.prior_predictive["x_obs"],
        #     kind="hist",
        #     hist_kwargs={"alpha": 0.6},
        #     label="simulated",
        # )
        # plt.xticks(rotation=45);

    logger.info("Marginalized mixture model with discrete z created (not sampled).")

    return model


"""Markov-Random Fields for spatial smoothing of priors"""


def mrf_regularization(
    data_scaled,
    idata,
    prior_init,
    x,
    y,
    *,
    n_neighbors=8,
    length_scale=None,
    beta=2.0,
    n_iter=5,
):
    """
    Mean-field Potts smoothing of priors:
      log π_i <- log π_i + β ∑_{j∈N(i)} w_ij q_j
      q_i ∝ exp(loglik_i + log π_i)
    Uses posterior means of μ/σ from idata (scaled space).
    Returns prior_final, q_final.
    """
    # posterior means (chains,draws,k,d) -> (k,d)
    mu = idata.posterior["mu"].mean(dim=["chain", "draw"]).values
    sigma = idata.posterior["sigma"].mean(dim=["chain", "draw"]).values
    if mu.ndim == 1:  # (K,) -> (K,1)
        mu = mu[:, None]
        sigma = sigma[:, None]

    W = build_knn_graph(x, y, n_neighbors=n_neighbors, length_scale=length_scale)
    pi = prior_init.copy()
    q = _responsibilities(data_scaled, mu, sigma, pi)

    for _ in range(n_iter):
        pi = _mrf_update(pi, q, W, beta)
        q = _responsibilities(data_scaled, mu, sigma, pi)

    return pi, q


def build_knn_graph(x, y, n_neighbors=8, length_scale=None):
    """
    Build symmetric kNN affinity W (csr). If length_scale is given (pixels),
    apply Gaussian weights exp(-d^2/(2*ls^2)); else use unit weights.
    """
    pts = np.column_stack([x, y]).astype(float)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(pts)
    dists, idx = nbrs.kneighbors(pts, return_distance=True)  # (N, k+1), col0=self

    N = pts.shape[0]
    rows = np.repeat(np.arange(N), n_neighbors)
    cols = idx[:, 1:].ravel()
    d = dists[:, 1:].ravel()

    if length_scale is None or length_scale <= 0:
        w = np.ones_like(d)
    else:
        w = np.exp(-(d**2) / (2.0 * (length_scale**2)))

    W = csr_matrix((w, (rows, cols)), shape=(N, N))
    return W.maximum(W.transpose())


def _gaussian_loglik(X, mu, sigma):
    """
    Log-likelihood under diagonal Gaussian, X:(N,D), mu/sigma:(K,D) -> (N,K)
    """
    xc = (X[:, None, :] - mu[None, :, :]) / (sigma[None, :, :] + EPS)
    return -0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma[None, :, :] + EPS) + xc**2).sum(
        axis=2
    )


def _responsibilities(X, mu, sigma, prior_probs):
    log_lik = _gaussian_loglik(X, mu, sigma)  # (N,K)
    log_prior = np.log(prior_probs + EPS)  # (N,K)
    logits = log_lik + log_prior
    a = logits.max(axis=1, keepdims=True)
    q = np.exp(logits - a)
    q /= q.sum(axis=1, keepdims=True)
    return q


def _mrf_update(prior_probs, q, W, beta):
    # message term from neighbors: (N,K)
    msg = W.dot(q)  # sparse @ dense
    logits = np.log(prior_probs + EPS) + beta * msg
    a = logits.max(axis=1, keepdims=True)
    pi = np.exp(logits - a)
    pi /= pi.sum(axis=1, keepdims=True)
    return pi
