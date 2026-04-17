from __future__ import annotations

import logging
from typing import Any

import numpy as np
import omegaconf
import pymc as pm
from pymc import math as pm_math
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("ppcx")

RANDOM_SEED = 8927
EPS = 1e-12
rng = np.random.default_rng(RANDOM_SEED)


""" Mixture models with spatial priors"""


def build_marginalized_mixture_model(
    data: np.ndarray,
    prior_probs: np.ndarray,
    sectors: dict[str, Any],
    mu_params: dict[str, Any],
    sigma_params: dict[str, Any],
    enforce_ordered_means: bool = False,
) -> pm.Model:
    """
    Build a marginalized Gaussian Mixture Model (GMM) in PyMC.

    This model marginalizes out the discrete cluster assignment variable 'z',
    treating the total log-probability as a `pm.Potential`. This formulation
    is often more robust for NUTS sampling than discrete latent variables.
    It supports JAX/NumPyro sampling (GPU acceleration).

    Args:
        data (np.ndarray):
            Observed data array of shape (n_obs, n_features).
            Usually scaled (e.g., StandardScaler).
        prior_probs (np.ndarray):
            Prior probabilities for each point belonging to each cluster,
            based on spatial location. Shape (n_obs, n_clusters).
        sectors (dict[str, Any]):
            Dictionary of sectors, used to determine the number of clusters (k).
        mu_params (dict[str, Any]):
            Hyperparameters for the means priors (e.g., {'mu': 0, 'sigma': 1}).
        sigma_params (dict[str, Any]):
            Hyperparameters for the standard deviation priors (e.g., {'sigma': 1}).
        enforce_ordered_means (bool, optional):
            If True, enforces that the means of the first feature (index 0) are ordered strictly (mu[0] < mu[1] < ... < mu[k]).
            This is critical for Anomaly Detection (Base < Anomaly) to prevent
            label switching, but likely harmful for generic spatial sector clustering. Defaults to False.

    Returns:
        pm.Model: A PyMC model object (compiled, but not sampled).
    """
    n_data = data.shape[0]
    n_features = data.shape[1]
    k = len(sectors)

    # Coordinates help PyMC organize output (InferenceData) dimensions
    model = pm.Model(
        coords={"obs": range(n_data), "cluster": range(k), "feature": range(n_features)}
    )
    with model:
        # ------------------------------------------------------------------
        # 1. Data Containers
        # ------------------------------------------------------------------
        # Using pm.Data allow us to change data values later (e.g. for
        # re-sampling) without recompiling the JAX computation graph.
        obs_data = pm.Data("obs_data", data, dims=("obs", "feature"))

        # Prior weights (spatial probabilities) -> Log-domain
        # Shape: (n_obs, n_clusters)
        prior_w = pm.Data(
            "prior_w", prior_probs.reshape(n_data, k), dims=("obs", "cluster")
        )
        log_w = pm_math.log(prior_w + 1e-12)  # Add epsilon to prevent log(0)

        # ------------------------------------------------------------------
        # 2. Priors for Cluster Parameters (Mu, Sigma)
        # ------------------------------------------------------------------
        # mu_params["mu"] and mu_params["sigma"] can be:
        #   - scalar (float/int): same prior for all clusters
        #   - 1-D array of length K: per-cluster prior
        # sigma_params["sigma"] follows the same convention.
        #
        # For mu_feat0 (shape (k,)): pass values as-is (scalar broadcasts, (k,) maps 1-to-1).
        # For sigma (shape (k, n_features)): we need (k, n_features) or scalar.
        #   If user gave a 1-D array of length K (per-cluster), reshape to (k, 1)
        #   so it broadcasts over the feature axis.

        mu_mu = mu_params["mu"].copy()  # scalar or array(k,)
        mu_sigma = mu_params["sigma"].copy()  # scalar or array(k,)
        sigma_val = sigma_params[
            "sigma"
        ].copy()  # scalar or array(k,) or array(k, n_features)

        if isinstance(mu_mu, omegaconf.listconfig.ListConfig):
            mu_mu = np.array(mu_mu)
        if isinstance(mu_sigma, omegaconf.listconfig.ListConfig):
            mu_sigma = np.array(mu_sigma)
        if isinstance(sigma_val, omegaconf.listconfig.ListConfig):
            sigma_val = np.array(sigma_val)

        # -- Parameters for the FIRST feature (assumed to be Velocity 'V') --
        if enforce_ordered_means:
            # Enforce mu[0] < mu[1] < ... for feature 0.
            # This breaks symmetry and prevents label switching in anomaly detection.
            # We use an ordered transform on the first feature's means.

            initval = None
            if np.isscalar(mu_mu):
                initval = np.linspace(mu_mu - 1, mu_mu + 1, k)
            elif (
                isinstance(mu_mu, np.ndarray)
                and mu_mu.ndim == 1
                and mu_mu.shape[0] == k
            ) or (isinstance(mu_mu, list) and len(mu_mu) == k):
                initval = np.sort(mu_mu)
            else:
                raise ValueError(
                    "Invalid mu_params['mu'] for ordered means. Must be scalar or 1-D array/list of length K."
                )

            mu_0 = pm.Normal(
                "mu_feat0",
                mu=mu_mu,
                sigma=mu_sigma,
                shape=(k,),
                transform=pm.distributions.transforms.ordered,
                # Initialization helper: spread means out to help constraint satisfaction
                initval=initval,
            )

        else:
            # Standard independent means for feature 0
            mu_0 = pm.Normal(
                "mu_feat0",
                mu=mu_mu,
                sigma=mu_sigma,
                shape=(k,),
            )

        # -- Handle remaining features (if any) --
        if n_features > 1:
            # Features 1..N are always independent (no ordering constraint)
            # mu_mu / mu_sigma may be shape (k,) (per-cluster scalars). PyMC
            # broadcasts a (k,) array as (1, k), which is incompatible with
            # shape=(k, n_features-1). Reshape to (k, 1) so it broadcasts correctly.
            mu_mu_rest = (
                mu_mu.reshape(k, 1)
                if isinstance(mu_mu, np.ndarray) and mu_mu.shape == (k,)
                else mu_mu
            )
            mu_sigma_rest = (
                mu_sigma.reshape(k, 1)
                if isinstance(mu_sigma, np.ndarray) and mu_sigma.shape == (k,)
                else mu_sigma
            )
            mu_rest = pm.Normal(
                "mu_rest",
                mu=mu_mu_rest,
                sigma=mu_sigma_rest,
                shape=(k, n_features - 1),
            )
            # Concatenate to form full mean matrix: (k, 1) + (k, n-1) -> (k, n)
            mu = pm.Deterministic(
                "mu",
                pm_math.concatenate([mu_0[:, None], mu_rest], axis=1),
                dims=("cluster", "feature"),
            )
        else:
            # Single feature case: (k,) -> (k, 1)
            mu = pm.Deterministic("mu", mu_0[:, None], dims=("cluster", "feature"))

        # -- Standard Deviations (HalfNormal) --
        # We assume diagonal covariance (features are independent given cluster)
        # Prepare sigma prior: if 1-D array(k,), reshape to (k,1) for broadcasting
        if (
            isinstance(sigma_val, np.ndarray)
            and sigma_val.ndim == 1
            and sigma_val.shape[0] == k
        ):
            sigma_val = sigma_val.reshape(k, 1)

        sigma = pm.HalfNormal("sigma", sigma_val, dims=("cluster", "feature"))

        # ------------------------------------------------------------------
        # 3. Likelihood Construction (Marginalized)
        # ------------------------------------------------------------------
        # We implement the log-likelihood manually to ensure shape broadcasting works correctly across (obs, cluster, feature) dimensions.

        # Standardize data relative to each cluster mean/sigma
        # Shape: (n_obs, n_clusters, n_features)
        x_centered = (obs_data[:, None, :] - mu[None, :, :]) / sigma[None, :, :]

        # Gaussian Log-Likelihood per feature
        # -0.5 * (log(2pi) + 2*log(sigma) + z^2)
        logp_feat = -0.5 * (
            pm_math.log(2 * np.pi) + 2 * pm_math.log(sigma[None, :, :]) + x_centered**2
        )

        # Sum log-probs across features (independent features assumption)
        # Shape: (n_obs, n_clusters)
        logp_clusters = logp_feat.sum(axis=2)

        # ------------------------------------------------------------------
        # 4. Mixture Log-Likelihood
        # ------------------------------------------------------------------
        # Compute log( sum( w_k * N(x | mu_k, sigma_k) ) )
        # logsumexp trick is used for numerical stability:
        # log(sum(exp(log_w + logp_cluster)))
        log_mix = pm.logsumexp(logp_clusters + log_w, axis=1)  # Shape: (n_obs,)

        # Add total log-likelihood to model as a Potential
        # (This is equivalent to observed=... but works effectively for marginalized models)
        pm.Potential("mixture_logp", log_mix.sum())

    logger.info(
        f"Marginalized mixture model created. Shape: (N={n_data}, K={k}, D={n_features}). "
        f"Ordered Means: {enforce_ordered_means}"
    )
    return model


def build_discrete_marginalized_mixture_model(
    data: np.ndarray,
    prior_probs: np.ndarray,
    sectors: dict[str, Any],
) -> pm.Model:
    """
    Build a mixture model with discrete latent variables for cluster assignment.

    WARNING: This model uses `pm.Categorical` for assignments (z). NUTS samplers
    (including JAX/NumPyro) generally struggle with discrete parameters because
    they cannot compute gradients. This function is provided for legacy comparison
    or for use with Gibbs/Metropolis samplers, but `build_marginalized_mixture_model`
    is strongly recommended for efficiency.

    Args:
        data (np.ndarray):
            Observed data array of shape (n_obs, n_features).
        prior_probs (np.ndarray):
            Prior probabilities for cluster assignments. Shape (n_obs, n_clusters).
        sectors (dict[str, Any]):
            Dictionary defining the sectors/clusters.

    Returns:
        pm.Model: A PyMC model with discrete latent variables.
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

        # Likelihood
        # Since 'z' is discrete, we index into mu/sigma using z.
        # This indexing breaks gradients for NUTS.
        pm.Normal(
            "obs",
            mu=mu[z],  # Select mean based on assignment z
            sigma=sigma[z],  # Select sigma based on assignment z
            observed=data,
            dims=("obs", "feature"),
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
    data_scaled: np.ndarray,
    idata: Any,
    prior_init: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int = 8,
    length_scale: float | None = None,
    beta: float = 2.0,
    n_iter: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Mean-Field approximation for MRF (Markov Random Field) spatial smoothing.

    This function refines the spatial prior probabilities based on the clustering
    results and the local neighborhood structure. It encourages spatially
    contiguous clusters by inspecting the posterior probabilities of neighbors.

    The update rule approximates a Potts model:
      log π_i <- log(prior_i) + β * ∑_{j ∈ Neighbors(i)} w_ij * q_j

    Where:
      - π_i: Updated spatial prior for point i.
      - q_j: Current responsibility (probability of cluster assignment) of neighbor j.
      - w_ij: Spatial weight (distance-based or binary) between i and j.
      - β (beta): Strength of the smoothing (inverse temperature).

    Args:
        data_scaled (np.ndarray):
            Feature data (N, D), scaled (e.g., StandardScaler).
        idata (Any):
            MCMC results containing posterior samples for 'mu' and 'sigma'.
            Used to compute the data likelihood term.
        prior_init (np.ndarray):
            Initial spatial priors (N, K).
        x (np.ndarray): X coordinates of points (N,).
        y (np.ndarray): Y coordinates of points (N,).
        n_neighbors (int, optional):
            Number of nearest neighbors to construct the spatial graph. Defaults to 8.
        length_scale (float | None, optional):
            Length scale for Gaussian kernel weights (exp(-d^2 / 2l^2)).
            If None, uses binary weights (1 for neighbors, 0 otherwise). Defaults to None.
        beta (float, optional):
            Smoothing strength. Higher values enforce stronger spatial continuity. Defaults to 2.0.
        n_iter (int, optional):
            Number of Mean-Field iteration steps. Defaults to 5.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - pi (N, K): The spatially smoothed prior probabilities.
            - q (N, K): The final responsibilities (posterior probabilities) after smoothing.
    """
    # 1. Extract cluster parameters (mu, sigma) from MCMC posterior
    # We use the mean of the posterior as a point estimate for the likelihood calculation
    # Shape: (chains, draws, k, d) -> mean -> (k, d)
    mu = idata.posterior["mu"].mean(dim=["chain", "draw"]).values
    sigma = idata.posterior["sigma"].mean(dim=["chain", "draw"]).values

    # Handle single cluster case dimensions: (K,) -> (K, 1)
    if mu.ndim == 1:
        mu = mu[:, None]
        sigma = sigma[:, None]

    # 2. Build Spatial Graph (Adjacency Matrix W)
    W = build_knn_graph(x, y, n_neighbors=n_neighbors, length_scale=length_scale)

    # 3. Initialize Iterative Smoothing
    pi = prior_init.copy()

    # Initial responsibilities 'q' based on Data Likelihood * Initial Prior
    q = _responsibilities(data_scaled, mu, sigma, pi)

    # 4. Mean-Field Iterations
    for i in range(n_iter):
        # Update priors 'pi' based on neighbors' responsibilities 'q'
        pi = _mrf_update(pi, q, W, beta)

        # Update responsibilities 'q' based on new priors 'pi'
        q = _responsibilities(data_scaled, mu, sigma, pi)

    return pi, q


def build_knn_graph(
    x: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 8,
    length_scale: float | None = None,
) -> csr_matrix:
    """
    Build a symmetric k-Nearest Neighbors (kNN) affinity matrix.

    Args:
        x (np.ndarray): X coordinates.
        y (np.ndarray): Y coordinates.
        n_neighbors (int): Number of neighbors per point.
        length_scale (float | None):
            If provided, weights are Gaussian: w = exp(-dist^2 / (2*scale^2)).
            If None, weights are binary (1.0).

    Returns:
        csr_matrix: Sparse symmetric adjacency matrix (N, N).
    """
    pts = np.column_stack([x, y]).astype(float)

    # Find k+1 neighbors because the first neighbor is the point itself (distance 0)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(pts)
    dists, idx = nbrs.kneighbors(pts, return_distance=True)

    # Prepare sparse matrix data (exclude self-loop at index 0)
    N = pts.shape[0]
    # Rows: [0, 0, ..., 1, 1, ...]
    rows = np.repeat(np.arange(N), n_neighbors)
    # Cols: Flattened neighbor indices
    cols = idx[:, 1:].ravel()
    # Dists: Flattened neighbor distances
    d = dists[:, 1:].ravel()

    # Compute weights
    if length_scale is None or length_scale <= 0:
        w = np.ones_like(d)
    else:
        # Gaussian kernel decaying with distance
        w = np.exp(-(d**2) / (2.0 * (length_scale**2)))

    # Construct sparse matrix
    W = csr_matrix((w, (rows, cols)), shape=(N, N))

    # Symmetrize the graph: w_ij = max(w_ij, w_ji)
    # This ensures if i is a neighbor of j, j is effectively linked to i.
    return W.maximum(W.transpose())


def _gaussian_loglik(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Compute log-likelihood of data X under a diagonal Gaussian model.

    Args:
        X: Data (N, D)
        mu: Means (K, D)
        sigma: Standard deviations (K, D)

    Returns:
        np.ndarray: Log-likelihoods (N, K)
    """
    # Broadcasting:
    # X: (N, 1, D)
    # mu: (1, K, D)
    # Result xc: (N, K, D) - Z-scores
    xc = (X[:, None, :] - mu[None, :, :]) / (sigma[None, :, :] + EPS)

    # Log PDF: -0.5 * (log(2pi) + 2*log(sigma) + z_score^2)
    # Sum over feature dimension D (axis 2) assuming independence
    return -0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma[None, :, :] + EPS) + xc**2).sum(
        axis=2
    )


def _responsibilities(
    X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, prior_probs: np.ndarray
) -> np.ndarray:
    """
    Compute posterior cluster probabilities (responsibilities).

    q(z=k | x) ∝ Likelihood(x | z=k) * Prior(z=k)
    """
    # 1. Calculate Log-Likelihood of data given Gaussian parameters
    # shape: (N, K)
    log_lik = _gaussian_loglik(X, mu, sigma)

    # 2. Add the Log-Prior probability (the spatial weight)
    # shape: (N, K)
    log_prior = np.log(prior_probs + EPS)

    # 3. Compute unnormalized log-posterior (logits)
    # log(Likelihood * Prior) = log(Likelihood) + log(Prior)
    logits = log_lik + log_prior

    # 4. Softmax (Numerical stability trick: subtract max)
    a = logits.max(axis=1, keepdims=True)
    q = np.exp(logits - a)
    q /= q.sum(axis=1, keepdims=True)  # Normalize to sum to 1

    return q


def _mrf_update(
    prior_probs: np.ndarray, q: np.ndarray, W: csr_matrix, beta: float
) -> np.ndarray:
    """
    Update spatial priors based on neighbor responsibilities.

    log π_new = log π_old + β * (AVG neighbors' q)
    """
    # Message passing: Sum of q from neighbors
    # W (N,N) sparse dot q (N,K) dense -> msg (N,K)
    msg = W.dot(q)

    # Update step in log domain
    # Existing Log Prior + Beta * Neighbor Influence
    logits = np.log(prior_probs + EPS) + beta * msg

    # Softmax normalization to get valid probabilities
    a = logits.max(axis=1, keepdims=True)
    pi = np.exp(logits - a)
    pi /= pi.sum(axis=1, keepdims=True)

    return pi
