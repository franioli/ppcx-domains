import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm as scipy_norm

logger = logging.getLogger("ppcx")
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

COLORMAP = plt.get_cmap("tab10")


"""POSTERIOR AND STATISTICS"""


def compute_posterior_assignments(
    idata: Any,
    *,
    n_posterior_samples: int | None = None,
    use_posterior_mean: bool = False,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute posterior responsibilities, hard labels, and uncertainty.

    Args:
        idata: ArviZ ``InferenceData`` from marginalized model (must contain variables
            ``mu`` and ``sigma`` in ``idata.posterior`` with dims ``(chain, draw, k, d)``).
        X_scaled: Array of shape ``(n_points, n_features)`` used for inference.
        prior_probs: Array of shape ``(n_points, k)`` with spatial priors for each point.
        n_posterior_samples: If provided, randomly subsample this many posterior draws
            to approximate responsibilities; otherwise use all draws.
        use_posterior_mean: If True, compute responsibilities using posterior mean
            of ``mu`` and ``sigma`` (faster, deterministic).
        random_seed: RNG seed used when subsampling posterior draws.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - posterior_probs: ``(n_points, k)`` averaged responsibilities.
            - cluster_pred: ``(n_points,)`` hard labels (argmax over clusters).
            - uncertainty: ``(n_points,)`` entropy of responsibilities per point.
    """
    # Extract observed data used for inference
    if "obs_data" not in idata.constant_data:
        raise ValueError(
            "compute_posterior_assignments: idata.constant_data must contain 'obs_data'."
        )
    obs_data = idata.constant_data["obs_data"].to_numpy()

    # Extract prior probabilities
    if "prior_w" not in idata.constant_data:
        raise ValueError(
            "compute_posterior_assignments: idata.constant_data must contain 'prior_w'."
        )
    prior_probs = idata.constant_data["prior_w"].to_numpy()

    # Extract posterior mu and sigma
    mu_samples = idata.posterior["mu"].values  # (chains, draws, k, n_features)
    sigma_samples = idata.posterior["sigma"].values

    # collapse chain/draw dims
    # S_full = mu_samples.shape[0] * mu_samples.shape[1]
    k = mu_samples.shape[2]
    n_features = mu_samples.shape[3]
    n_points = obs_data.shape[0]

    mu_flat = mu_samples.reshape(-1, k, n_features)  # (S_full, k, d)
    sigma_flat = sigma_samples.reshape(-1, k, n_features)

    if use_posterior_mean:
        # Fast deterministic assignment using posterior mean parameters
        mu_mean = mu_flat.mean(axis=0)  # (k, d)
        sigma_mean = sigma_flat.mean(axis=0)
        # compute log-likelihood for each cluster (vectorized)
        # shape -> (n_points, k)
        log_lik = np.stack(
            [
                scipy_norm.logpdf(obs_data, loc=mu_mean[kk], scale=sigma_mean[kk]).sum(
                    axis=1
                )
                for kk in range(k)
            ],
            axis=1,
        )
        log_prior = np.log(prior_probs + 1e-12)  # (n_points, k)
        log_resp = log_prior + log_lik
        # normalize
        a = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - a)
        resp /= resp.sum(axis=1, keepdims=True)
        posterior_probs = resp
    else:
        # Monte Carlo average over posterior draws (may be expensive)
        rng = np.random.default_rng(random_seed)
        if n_posterior_samples is None or n_posterior_samples >= mu_flat.shape[0]:
            sel_idx = np.arange(mu_flat.shape[0])
        else:
            sel_idx = rng.choice(
                mu_flat.shape[0], size=n_posterior_samples, replace=False
            )

        S = sel_idx.shape[0]
        # accumulate responsibilities per draw
        resp_acc = np.zeros((S, n_points, k), dtype=float)
        for si, s in enumerate(sel_idx):
            # vectorized over points and features:
            # for each component kk compute logpdf across features and sum
            for kk in range(k):
                lp = scipy_norm.logpdf(
                    obs_data, loc=mu_flat[s, kk], scale=sigma_flat[s, kk]
                )  # (n_points, d)
                log_lik = lp.sum(axis=1)  # (n_points,)
                log_prior = np.log(prior_probs[:, kk] + 1e-12)  # (n_points,)
                resp_acc[si, :, kk] = log_prior + log_lik
            # stabilize & normalize for this draw
            a = resp_acc[si].max(axis=1, keepdims=True)
            resp_acc[si] = np.exp(resp_acc[si] - a)
            resp_acc[si] /= resp_acc[si].sum(axis=1, keepdims=True)

        # average over selected draws
        posterior_probs = resp_acc.mean(axis=0)  # (n_points, k)

    # hard assignment, uncertainty (entropy)
    cluster_pred = posterior_probs.argmax(axis=1)
    uncertainty = -np.sum(posterior_probs * np.log(posterior_probs + 1e-12), axis=1)

    return posterior_probs, cluster_pred, uncertainty


def compute_entropy(posterior_probs: np.ndarray) -> np.ndarray:
    """Compute per-point entropy from posterior probabilities.

    Args:
        posterior_probs: Array ``(n_points, k)`` with responsibilities.
    Returns:
        np.ndarray: Entropy per point, shape ``(n_points,)``.
    """
    return -np.sum(posterior_probs * np.log(posterior_probs + 1e-12), axis=1)


def compute_max_probs(
    posterior_probs: np.ndarray, cluster_pred: np.ndarray
) -> np.ndarray:
    """Compute per-point maximum posterior probability from cluster assignments.

    Args:
        posterior_probs: Array ``(n_points, k)`` with responsibilities.
        cluster_pred: Array ``(n_points,)`` with hard labels per point.
    Returns:
        np.ndarray: Maximum posterior probability per point, shape ``(n_points,)``.
    """
    return posterior_probs[np.arange(len(cluster_pred)), cluster_pred]


def get_model_parameters_from_idata(
    idata: az.InferenceData,
    scaler: Any | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract model parameters from ArviZ InferenceData object.

    Args:
        idata: ArviZ ``InferenceData`` containing posterior draws for ``mu`` and ``sigma``.
        scaler: Optional scaler to inverse-transform the selected feature (applied to model parameters if provided).
        feature_index: Feature index to report ``μ/σ`` for (default ``0``).

    Returns:
        Tuple[np.ndarray | None, np.ndarray | None]: Arrays ``(k,)`` with posterior means for ``mu`` and ``sigma`` for the selected feature, or ``None`` if not available.
    """

    if "mu" not in idata.posterior or "sigma" not in idata.posterior:  # type: ignore
        logger.error(
            "InferenceData does not contain 'mu' or 'sigma' in posterior. Cannot extract model parameters."
        )
        return None, None

    mu_arr = idata.posterior["mu"].mean(dim=["chain", "draw"]).values  # type: ignore
    sigma_arr = idata.posterior["sigma"].mean(dim=["chain", "draw"]).values  # type: ignore

    # Ensure shapes (k, n_features)
    if mu_arr.ndim == 1:
        mu_arr = mu_arr[:, None]
        sigma_arr = sigma_arr[:, None]

    # Optionally inverse-transform
    if scaler is not None:
        # Ensure the number of features matches with scaler: expects shape (n_samples, n_features)
        n_features = mu_arr.shape[1]
        if n_features != scaler.scale_.shape[0]:
            logger.error(
                f"Model parameters have {n_features} features, but scaler expects {scaler.scale_.shape[0]}."
            )
            return None, None

        # Inverse-transform μ
        mu_arr = scaler.inverse_transform(mu_arr)

        # Inverse-transform σ with feature scale (Robust/Standard scalers expose scale_)
        scale_vec = getattr(scaler, "scale_", None)
        if scale_vec is not None:
            sigma_arr = sigma_arr * scale_vec

    return mu_arr, sigma_arr


def compute_cluster_statistics(
    *,
    df_features: pd.DataFrame,
    cluster_pred: np.ndarray,
    posterior_probs: np.ndarray,
    idata: Any | None = None,
    scaler: Any | None = None,
) -> dict[int, dict[str, float]]:
    """
    Compute per-cluster statistics independently of plotting.

    Args:
        df_features: DataFrame with columns ``x``, ``y``, and ``V`` (velocity magnitude).
        cluster_pred: Array ``(n_points,)`` with hard labels per point.
        posterior_probs: Array ``(n_points, k)`` with responsibilities.
        idata: Optional ArviZ ``InferenceData`` to compute model posterior means for
            ``mu`` and ``sigma`` (reported as ``model_mu``/``model_sigma`` for the
            selected feature).
        scaler: Optional scaler to inverse-transform the selected feature (applied to
            model parameters if provided).
        feature_index: Feature index to report ``μ/σ`` for (default ``0``).

    Returns:
        Dict[int, Dict[str, float]]: Mapping cluster_id to statistics, including keys:
            ``count``, ``x_mean``, ``y_mean``, ``x_std``, ``y_std``,
            ``velocity_mean``, ``velocity_std``, ``velocity_median``, ``velocity_nmad``,
            ``avg_entropy``, ``avg_assignment_prob``, ``model_mu``, ``model_sigma``.
    """
    # Per-point entropy_pt and max posterior prob
    entropy_pt = compute_entropy(posterior_probs)
    max_prob_pt = compute_max_probs(posterior_probs, cluster_pred)

    # Optional model μ/σ from posterior means
    if idata is not None:
        model_mu, model_sigma = get_model_parameters_from_idata(idata, scaler)
    else:
        model_mu, model_sigma = None, None

    velocity = df_features["V"].to_numpy()
    stats: dict[int, dict[str, float]] = {}
    for i, label in enumerate(np.unique(cluster_pred)):
        mask = cluster_pred == label
        count = int(mask.sum())
        if count == 0:
            continue

        v_vals = velocity[mask]
        v_mean = float(v_vals.mean())
        v_std = float(v_vals.std())
        v_median = float(np.median(v_vals))
        nmad = float(np.median(np.abs(v_vals - v_median)) * 1.4826)

        x_vals = np.asarray(df_features.loc[mask, "x"])  # Series -> ndarray
        y_vals = np.asarray(df_features.loc[mask, "y"])  # Series -> ndarray
        x_mean = float(x_vals.mean())
        y_mean = float(y_vals.mean())
        x_std = float(x_vals.std())
        y_std = float(y_vals.std())
        avg_entropy = float(entropy_pt[mask].mean())
        avg_prob = float(max_prob_pt[mask].mean())

        # Model parameters for the selected cluster
        if model_mu is not None and model_sigma is not None:
            n_model_clusters = model_mu.shape[0]

            if int(label) >= n_model_clusters:
                logger.warning(
                    f"Cluster label {label} exceeds number of clusters in model ({n_model_clusters})."
                )
                model_mu_val = None
                model_sigma_val = None
            elif model_mu.shape[1] > 1:
                # Multi-feature case: return the vector for the cluster
                model_mu_val = model_mu[int(label), :]
                model_sigma_val = model_sigma[int(label), :]
            else:
                # Single-feature case: use [label, 0] to get the scalar element correctly
                model_mu_val = float(model_mu[int(label), 0])
                model_sigma_val = float(model_sigma[int(label), 0])
        else:
            model_mu_val = None
            model_sigma_val = None

        entry = {
            "count": count,
            "x_mean": x_mean,
            "y_mean": y_mean,
            "x_std": x_std,
            "y_std": y_std,
            "velocity_mean": v_mean,
            "velocity_std": v_std,
            "velocity_median": v_median,
            "velocity_nmad": nmad,
            "avg_entropy": avg_entropy,
            "avg_assignment_prob": avg_prob,
            "model_mu": model_mu_val,
            "model_sigma": model_sigma_val,
        }

        stats[int(label)] = entry

    return stats


# === Deprecated functions for metadata collection and saving ===#


def collect_run_metadata(
    idata: az.InferenceData,
    convergence_flag: bool,
    data_array_scaled: np.ndarray,
    variables_names: list,
    sectors: dict,
    prior_probs: np.ndarray,
    sample_args: dict,
    **kwargs,
) -> dict:
    """Automatically collect metadata from current variables and context."""

    logger.warning(
        f"Function {collect_run_metadata.__name__} is temporary and it will be replaced by a more structured configuration system."
    )

    # Get variables from current namespace/globals
    frame = kwargs.get("frame", globals())

    metadata = {
        "experiment": {
            "name": frame.get("base_name", "unknown_experiment"),
            "timestamp": datetime.now().isoformat(),
            "random_seed": frame.get("RANDOM_SEED", None),
        },
        "data": {
            "camera_name": frame.get("camera_name"),
            "reference_start_date": frame.get("reference_start_date"),
            "reference_end_date": frame.get("reference_end_date"),
            "dt_min_hours": frame.get("dt_min"),
            "dt_max_hours": frame.get("dt_max"),
            "subsample_factor": frame.get("SUBSAMPLE_FACTOR"),
            "subsample_method": frame.get("SUBSAMPLE_METHOD"),
            "filter_kwargs": frame.get("filter_kwargs"),
            "roi_path": str(frame.get("roi_path", "")),
            "sector_prior_file": str(frame.get("SECTOR_PRIOR_FILE", "")),
            "multiscale": "sigma_values" in frame,
            "gaussian_smoothing_scales": frame.get("sigma_values"),
            "n_observations": data_array_scaled.shape[0],
            "n_dic_analyses": len(frame.get("dic_ids", [])),
        },
        "model": {
            "type": "marginalized_mixture",
            "n_clusters": len(sectors),
            "n_features": data_array_scaled.shape[1],
            "feature_names": variables_names,
            "prior_specification": "spatial_sectors",
            "sectors": list(sectors.keys()),
            "prior_probabilities": frame.get("PRIOR_PROBABILITY"),
            "prior_shape": prior_probs.shape,
        },
        "sampling": sample_args,
        "convergence": {
            "converged": convergence_flag,
            "summary_stats": az.summary(idata, var_names=["mu", "sigma"]).to_dict(),
        },
    }

    return metadata


def save_run_metadata(
    output_dir: Path, base_name: str, metadata: dict, suffix: str = ""
):
    """Save metadata JSON with optional suffix."""
    metadata_file = output_dir / f"{base_name}_metadata{suffix}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Experiment metadata saved to {metadata_file}")
