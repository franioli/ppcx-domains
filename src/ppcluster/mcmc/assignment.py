import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image
from scipy.stats import mode
from scipy.stats import norm as scipy_norm
from sklearn.metrics import adjusted_rand_score

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
            if model_mu.shape[0] != len(np.unique(cluster_pred)):
                logger.warning(
                    "Number of clusters in model parameters does not match number of unique cluster labels."
                )
                model_mu_val = None
                model_sigma_val = None
            elif model_mu.shape[1] > 1:
                model_mu_val = model_mu[i, :]
                model_sigma_val = model_sigma[i, :]
            else:
                model_mu_val = float(model_mu[i])
                model_sigma_val = float(model_sigma[i])
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


def plot_velocity_clustering(
    df_features: pd.DataFrame,
    img: Image.Image | np.ndarray | None,
    *,
    idata: Any,
    cluster_pred: np.ndarray,
    posterior_probs: np.ndarray,
    scaler: Any | None = None,
) -> Figure:
    """Plot 1D velocity clustering results for marginalized model.

    Args:
        df_features: DataFrame with columns ``x``, ``y``, ``u``, ``v``, and ``V``.
        img: Optional background image array.
        idata: ArviZ ``InferenceData`` containing posterior draws for ``mu`` and ``sigma``.
        cluster_pred: Array of hard cluster assignments per point.
        posterior_probs: Array of responsibilities per point and cluster.
        scaler: Optional scaler to inverse-transform model parameters for overlay.

    Returns:
        Figure: Matplotlib figure containing the 4 subplots.
    """
    # Distinct colors
    unique_labels = np.unique(cluster_pred)

    # USE DEFAULT COLORMAP
    colors = [COLORMAP(i) for i in range(len(unique_labels))]
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    try:
        # Plot 1: Velocity field with quiver plot
        ax0 = axes[0, 0]
        ax0.set_title("Velocity Vector Field", fontsize=14, pad=10)
        if img is not None:
            ax0.imshow(img, alpha=0.5, cmap="gray")
        magnitudes = df_features["V"].to_numpy()
        vmin = 0.0
        vmax = np.max(magnitudes)
        norm = Normalize(vmin=vmin, vmax=vmax)
        q = ax0.quiver(
            df_features["x"].to_numpy(),
            df_features["y"].to_numpy(),
            df_features["u"].to_numpy(),
            df_features["v"].to_numpy(),
            magnitudes,
            scale=None,
            scale_units="xy",
            angles="xy",
            cmap="viridis",
            norm=norm,
            width=0.008,
            headwidth=2.5,
            alpha=1.0,
        )
        cbar = fig.colorbar(q, ax=ax0, shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label("Velocity Magnitude", rotation=270, labelpad=15)
        ax0.set_aspect("equal")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.grid(False)

        # Plot 2: Spatial clusters
        ax1 = axes[0, 1]
        ax1.set_title("Velocity-Based Spatial Clustering", fontsize=14, pad=10)
        if img is not None:
            ax1.imshow(img, alpha=0.3, cmap="gray")
        for label in unique_labels:
            mask = cluster_pred == label
            if np.any(mask):
                ax1.scatter(
                    df_features.loc[mask, "x"],
                    df_features.loc[mask, "y"],
                    c=color_map[label],
                    s=8,
                    alpha=0.8,
                    label=f"Cluster {label}",
                    edgecolors="none",
                )
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=10)
        ax1.set_aspect("equal")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot 3: Per-point uncertainty (entropy)
        entropy = compute_entropy(posterior_probs)
        ax2 = axes[1, 0]
        ax2.set_title("Assignment Uncertainty (Entropy)", fontsize=14, pad=10)
        vmin = 0.0
        vmax = max(entropy.max(), 0.1)  # Ensure non-zero range
        norm = Normalize(vmin=vmin, vmax=vmax)
        if img is not None:
            ax2.imshow(img, alpha=0.3, cmap="gray")
        scatter = ax2.scatter(
            df_features["x"],
            df_features["y"],
            c=entropy,
            cmap="plasma",
            s=8,
            alpha=0.8,
            norm=norm,
        )
        ax2.set_aspect("equal")
        ax2.set_xticks([])
        ax2.set_yticks([])
        cbar2 = plt.colorbar(scatter, ax=ax2, shrink=0.8)
        cbar2.set_label("Entropy", rotation=270, labelpad=20)

        # Plot 4: Velocity distribution by cluster
        ax3 = axes[1, 1]
        ax3.set_title("Velocity Distribution by Cluster", fontsize=14, pad=10)
        velocity = df_features["V"].values
        for label in unique_labels:
            mask = cluster_pred == label
            if np.any(mask):
                ax3.hist(
                    velocity[mask],
                    bins=35,
                    alpha=0.7,
                    density=True,
                    color=color_map[label],
                    label=f"Cluster {label}",
                    edgecolor="white",
                    linewidth=0.5,
                )

        # Overlay model distributions
        # Get model parameters for overlay (posterior means)
        mu_posterior, sigma_posterior = get_model_parameters_from_idata(
            idata, scaler=scaler
        )
        if mu_posterior is None or sigma_posterior is None:
            raise ValueError(
                "Model parameters not available in InferenceData for overlay."
            )

        # Group model parameters by cluster label
        if mu_posterior.shape[0] != len(unique_labels):
            raise ValueError(
                "Number of model clusters does not match number of unique labels. Cannot overlay model distributions."
            )

        mu_posterior = {
            label: mu_posterior[i, 0] for i, label in enumerate(unique_labels)
        }
        sigma_posterior = {
            label: sigma_posterior[i, 0] for i, label in enumerate(unique_labels)
        }

        # Ensure numpy array for range computation
        velocity_arr = np.asarray(velocity)
        v_range = np.linspace(
            float(np.min(velocity_arr)), float(np.max(velocity_arr)), 200
        )
        for label in unique_labels:
            model_dist = scipy_norm.pdf(
                v_range,
                mu_posterior[label],
                sigma_posterior[label],
            )
            ax3.plot(
                v_range,
                model_dist,
                "--",
                color=color_map[label],
                linewidth=2.5,
                alpha=0.9,
                label=f"Model {label}",
            )
        ax3.set_xlabel("Velocity Magnitude", fontsize=12)
        ax3.set_ylabel("Density", fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10, framealpha=0.9)

        # Render statistics box
        stats = compute_cluster_statistics(
            df_features=df_features,
            cluster_pred=cluster_pred,
            posterior_probs=posterior_probs,
            idata=idata,
            scaler=scaler,
        )
        if not stats:
            raise ValueError("Unable to compute cluster statistics (no clusters?)")

        fig.subplots_adjust(right=0.98)
        stats_text = "CLUSTER STATISTICS\n" + "=" * 40 + "\n"
        for label in sorted(stats.keys()):
            s = stats[label]
            stats_text += f"CLUSTER {label} (pts: {s['count']})\n"
            stats_text += f"├─ Center: ({s['x_mean']:.1f}, {s['y_mean']:.1f})\n"
            stats_text += f"├─ Spread: (σx={s['x_std']:.1f}, σy={s['y_std']:.1f})\n"
            stats_text += (
                f"├─ Velocity: {s['velocity_mean']:.4f} ± {s['velocity_std']:.4f}\n"
            )
            stats_text += (
                f"├─ Median/NMAD: {s['velocity_median']:.4f}/{s['velocity_nmad']:.4f}\n"
            )
            if s["model_mu"] is not None and s["model_sigma"] is not None:

                def _fmt(v):
                    arr = np.asarray(v)
                    if arr.size == 1:
                        return f"{float(arr.item()):.4f}"
                    return ", ".join(f"{float(x):.4f}" for x in arr.ravel())

                model_mu_str = _fmt(s["model_mu"])
                model_sigma_str = _fmt(s["model_sigma"])
                stats_text += f"├─ Model μ: {model_mu_str}\n"
                stats_text += f"├─ Model σ: {model_sigma_str}\n"

            stats_text += f"├─ Avg Entropy: {s['avg_entropy']:.4f}\n"
            stats_text += f"└─ Avg Assignment Prob: {s['avg_assignment_prob']:.4f}\n\n"

        fig.text(
            0.72,
            0.05,
            stats_text,
            fontsize=8,
            verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        )
    except Exception as exc:
        logger.error(f"Error generating clustering plots: {exc}")
    finally:
        plt.tight_layout(rect=(0, 0, 0.7, 1))

    return fig


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
