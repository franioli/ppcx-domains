import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image
from scipy.stats import mode
from scipy.stats import norm as scipy_norm
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from ppcluster.mcmc.assignment import (
    compute_cluster_statistics,
    compute_entropy,
    compute_posterior_assignments,
    get_model_parameters_from_idata,
)
from ppcluster.mcmc.models import (
    build_marginalized_mixture_model,
    mrf_regularization,
)
from ppcluster.mcmc.priors import plot_spatial_priors

logger = logging.getLogger("ppcx")

COLORMAP = plt.get_cmap("tab10")


@dataclass
class ClusteringResult:
    idata: az.InferenceData
    convergence_flag: bool
    posterior_probs: np.ndarray
    cluster_pred: np.ndarray
    uncertainty: np.ndarray
    priors: np.ndarray
    entropy: np.ndarray | None = None


def sample_model(
    model: pm.Model,
    output_dir: Path | None = None,
    base_name: str | None = None,
    sigma: float | int | None = None,
    force_cpu: bool = False,
    **kwargs,
) -> tuple[az.InferenceData, bool]:
    """
    Robust wrapper to sample a PyMC model.
    Attempts to use JAX/NumPyro on GPU by default unless force_cpu=True.
    Falls back to standard CPU sampling on any failure or if GPU is unavailable.
    """
    idata = None
    use_jax_attempt = False

    # 1. Check if we can and should attempt GPU sampling
    if not force_cpu:
        try:
            import jax

            # Check if JAX actually sees a GPU
            try:
                gpu_devices = jax.devices("gpu")
                if len(gpu_devices) > 0:
                    use_jax_attempt = True
                else:
                    logger.info("JAX installed, but no GPU device found. Using CPU.")
            except RuntimeError:
                logger.info(
                    "JAX installed, but GPU backend not initialized. Using CPU."
                )

        except ImportError:
            logger.info("NumPyro/JAX not installed. Using CPU.")

    n_data = len(model.coords["obs"])
    if n_data < 500 and not force_cpu:
        logger.info(
            f"Data size {n_data} is small; skipping JAX JIT overhead and using CPU."
        )
        use_jax_attempt = False

    # 2. Try JAX/GPU Sampling
    if use_jax_attempt:
        try:
            logger.info("Attempting JAX/GPU sampling (numpyro) in vectorized mode...")

            import jax

            # Use float32 for ~2x GPU speedup
            jax.config.update("jax_enable_x64", False)

            # Prepare GPU-specific arguments
            # We copy kwargs to avoid modifying them for the CPU fallback if this fails
            jax_kwargs = kwargs.copy()

            # Remove CPU-multiprocessing args that confuse JAX
            jax_kwargs.pop("cores", None)
            jax_kwargs.pop("mp_ctx", None)
            jax_kwargs.pop("progressbar_theme", None)

            # Optimization: Cap chains for VRAM safety on single GPU
            # 4 chains is usually sweet spot for efficiency vs diagnostics
            requested_chains = jax_kwargs.get("chains", 4)
            jax_kwargs["chains"] = min(requested_chains, 4)

            with model:
                idata = pm.sample(
                    nuts_sampler="numpyro",
                    nuts_sampler_kwargs={"chain_method": "vectorized"},
                    **jax_kwargs,
                )
            logger.info("JAX/GPU sampling completed successfully.")

        except Exception as e:
            logger.error(f"JAX/GPU sampling failed with error: {e}")
            logger.warning("Falling back to standard CPU sampling...")
            idata = None  # Ensure we trigger the fallback

    # 3. CPU Fallback (or if forced)
    if idata is None:
        mode_str = "Forced CPU" if force_cpu else "CPU Fallback"
        logger.info(f"Starting standard PyMC sampling ({mode_str})...")

        # Remove any GPU-specific args that might have been added for the JAX attempt
        cpu_kwargs = kwargs.copy()
        cpu_kwargs.pop("chain_method", None)
        cpu_kwargs.pop("init", None)
        with model:
            idata = pm.sample(**cpu_kwargs)
        logger.info("CPU sampling completed.")

    # 4. Convergence Checks
    has_converged = True
    try:
        # Basic check: R-hat < 1.05 is a common threshold
        rhat_max = az.rhat(idata).max().to_array().item()
        if rhat_max > 1.05:
            has_converged = False
            logger.warning(f"Convergence warning: Max R-hat is {rhat_max:.3f} (> 1.05)")
    except Exception:
        # If R-hat fails (e.g. single chain), assume converged or handle differently
        logger.debug("Could not compute R-hat (possibly single chain).")
        pass

    return idata, has_converged


def clusterize_gaussian_mixture(
    data_array_scaled: np.ndarray,
    prior_probs: np.ndarray,
    sectors: dict[str, Any],
    mu_params: dict | None = None,
    sigma_params: dict | None = None,
    sample_args: dict | None = None,
    apply_mrf_regularization: bool = False,
    x_pos: np.ndarray | None = None,
    y_pos: np.ndarray | None = None,
    mrf_kwargs: dict[str, Any] | None = None,
    second_pass: str = "short",
    second_pass_sample_args: dict | None = None,
    force_cpu: bool = False,
    random_seed: int = 8927,
) -> ClusteringResult:
    """
    Run MCMC-based clustering on preprocessed features with a Gaussian mixture model and optional MRF regularization.

    Args:
        data_array_scaled (np.ndarray): Scaled feature array for clustering.
        prior_probs (np.ndarray): Prior probabilities array (n_samples, n_clusters).
        sectors (dict): Dictionary of sector names to shapely Polygons.
        mu_params (dict, optional): Parameters for the mean of the mixture components. Default is None.
        sigma_params (dict, optional): Parameters for the standard deviation of the mixture components. Default is None.
        sample_args (dict, optional): Arguments for PyMC sampling. Default is None.
        apply_mrf_regularization (bool, optional): Whether to apply MRF regularization to spatial priors. Default is False.
        mrf_kwargs (dict, optional): Arguments for MRF regularization. Default is None.
        second_pass (str, optional): Strategy for second sampling pass ("skip", "short", "full"). Default is "full".
        second_pass_sample_args (dict, optional): Arguments for second pass sampling. Default is None.
        force_cpu (bool, optional): Whether to force CPU sampling even if GPU is available. Default is False.
        random_seed (int, optional): Random seed for reproducibility. Default is 8927.

    Returns:
        ClusteringResult: Dataclass containing results:
            - "idata": ArviZ InferenceData object
            - "convergence_flag": bool
            - "posterior_probs": np.ndarray
            - "cluster_pred": np.ndarray
            - "uncertainty": np.ndarray
            - "priors": np.ndarray
    """

    def _initvals_from_idata(idata_in, n_chains):
        mu_mean = idata_in.posterior["mu"].mean(dim=["chain", "draw"]).values
        sigma_mean = idata_in.posterior["sigma"].mean(dim=["chain", "draw"]).values
        init = {"mu": mu_mean, "sigma": sigma_mean}
        return [init for _ in range(n_chains)]

    logger.info("Running MCMC clustering...")

    if mu_params is None:
        mu_params = {"mu": 0, "sigma": 1}
    if sigma_params is None:
        sigma_params = {"sigma": 1}
    if sample_args is None:
        sample_args = dict(
            target_accept=0.95,
            draws=2000,
            tune=1000,
            chains=4,
            cores=4,
            random_seed=random_seed,
        )

    # Initializations
    priors = prior_probs.copy()

    # Build model
    model = build_marginalized_mixture_model(
        data_array_scaled,
        priors,
        sectors,
        mu_params=mu_params,
        sigma_params=sigma_params,
    )

    # Sample model (1st pass)
    idata, convergence_flag = sample_model(
        model, None, None, force_cpu=force_cpu, **sample_args
    )
    if not convergence_flag:
        idata_summary = az.summary(idata, var_names=["mu", "sigma"])
        logger.info(f"MCMC did not converge. Summary:\n{idata_summary}")

    # --- MRF regularization of priors and optional re-sample ---
    if apply_mrf_regularization:
        if x_pos is None or y_pos is None:
            raise ValueError("x_pos and y_pos must be provided for MRF regularization.")

        if mrf_kwargs is None:
            mrf_kwargs = {
                "n_neighbors": 8,
                "length_scale": 50,
                "beta": 2,
                "n_iter": 5,
            }
        logger.info("Applying MRF regularization to spatial priors...")
        priors, q_mrf = mrf_regularization(
            data_array_scaled, idata, priors, x_pos, y_pos, **mrf_kwargs
        )
        # try:
        #     fig, _ = plot_spatial_priors(df_input, prior_mrf, img=img)
        #     fig.savefig(
        #         output_dir
        #         / f"{base_name}_mrf_priors_neig{mrf_kwargs['n_neighbors']}_ls{mrf_kwargs['length_scale']}_beta{mrf_kwargs['beta']}.jpg",
        #         dpi=150,
        #         bbox_inches="tight",
        #     )
        #     plt.close(fig)
        # except Exception as exc:
        #     logger.warning(f"Could not plot MRF priors: {exc}")

        if second_pass.lower() == "skip":
            logger.info(
                "Skipping re-sampling after MRF regularization. Using pre-sampled MCMC posteriors."
            )
            posterior_probs = q_mrf
            cluster_pred = np.argmax(posterior_probs, axis=1)
            uncertainty = 1.0 - posterior_probs.max(axis=1)
            return ClusteringResult(
                idata=idata,
                convergence_flag=convergence_flag,
                posterior_probs=posterior_probs,
                cluster_pred=cluster_pred,
                uncertainty=uncertainty,
                priors=priors,
            )
        else:
            logger.info("Re-sampling with MRF-regularized priors...")
            with model:
                pm.set_data({"prior_w": priors})

            sp2_args = dict(**sample_args)
            if second_pass.lower() == "short":
                sp2_args.update(dict(draws=600, tune=400, chains=2, cores=2))
                if second_pass_sample_args:
                    sp2_args.update(second_pass_sample_args)
            elif second_pass_sample_args:
                sp2_args.update(second_pass_sample_args)

            initvals = _initvals_from_idata(idata, sp2_args.get("chains", 2))

            with model:
                idata, convergence_flag = sample_model(
                    model,
                    None,
                    None,
                    initvals=initvals,
                    force_cpu=force_cpu,
                    **sp2_args,
                )
            logger.info("Second pass MCMC clustering completed.")

    # Compute posterior-based assignments
    posterior_probs, cluster_pred, uncertainty = compute_posterior_assignments(
        idata, n_posterior_samples=200
    )

    # Compute entropy as an additional uncertainty measure
    entropy = -np.sum(posterior_probs * np.log(posterior_probs + 1e-10), axis=1)

    result = ClusteringResult(
        convergence_flag=convergence_flag,
        idata=idata,
        posterior_probs=posterior_probs,
        cluster_pred=cluster_pred,
        uncertainty=uncertainty,
        priors=priors,
        entropy=entropy,
    )

    return result


def save_sampling_summary(
    convergence_flag,
    idata,
    output_dir,
    base_name,
    make_plots=False,
    df_input: pd.DataFrame | None = None,
    cluster_pred: np.ndarray | None = None,
    posterior_probs: np.ndarray | None = None,
    scaler: StandardScaler | None = None,
    img: np.ndarray | None = None,
):
    sampling_info = {
        "convergence": convergence_flag,
        "summary_stats": az.summary(idata, var_names=["mu", "sigma"]).to_dict(),
    }
    with open(output_dir / f"{base_name}_sampling_summary.json", "w") as f:
        json.dump(sampling_info, f, indent=4)

    if make_plots:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        az.plot_trace(
            idata, var_names=["mu", "sigma"], axes=axes, compact=True, legend=True
        )
        fig.savefig(output_dir / f"{base_name}_trace_plots.jpg", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        az.plot_forest(
            idata, var_names=["mu", "sigma"], combined=True, ess=True, ax=axes
        )
        fig.savefig(output_dir / f"{base_name}_forest_plot.jpg", dpi=150)
        plt.close(fig)

    if (
        make_plots
        and df_input is not None
        and cluster_pred is not None
        and posterior_probs is not None
        and scaler is not None
    ):
        fig = plot_velocity_clustering(
            df_features=df_input,
            img=img,
            idata=idata,
            cluster_pred=cluster_pred,
            posterior_probs=posterior_probs,
            scaler=scaler,
        )
        fig.savefig(
            output_dir / f"{base_name}_results.jpg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


# --- Multiscale clustering aggregation ---
def aggregate_multiscale_clustering(
    results, similarity_threshold=0.6, overall_threshold=0.7, fig_path=None
):
    """
    Aggregate clustering results across scales, filtering unstable scales.

    Parameters:
    -----------
    results : list of dict
        Results from different scale clustering runs
    similarity_threshold : float
        Minimum mean similarity for a scale to be included
    overall_threshold : float
        Minimum overall similarity across scales to accept results

    Returns:
    --------
    combined_cluster_pred : ndarray
        Aggregated cluster assignments
    stability_score : float
        Measure of overall stability (0-1)
    """

    # Extract all cluster predictions
    all_cluster_preds = np.array([res["cluster_pred"] for res in results])
    n_scales = len(all_cluster_preds)
    sigma_values = [res["sigma"] for res in results]

    # Calculate pairwise similarities
    similarity_matrix = np.zeros((n_scales, n_scales))
    np.fill_diagonal(similarity_matrix, 1.0)
    for i, j in combinations(range(n_scales), 2):
        sim = adjusted_rand_score(all_cluster_preds[i], all_cluster_preds[j])
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim

    # Plot similarity heatmap
    if fig_path is not None:
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=sigma_values,
            yticklabels=sigma_values,
        )
        plt.title("Adjusted Rand Index Between Scales")
        plt.xlabel("Sigma")
        plt.ylabel("Sigma")
        plt.tight_layout()
        fig.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Calculate mean similarity for each scale (exclude self-similarity)
    mean_similarities = (similarity_matrix.sum(axis=1) - 1) / (n_scales - 1)

    # Filter scales with low similarity
    valid_scales = mean_similarities >= similarity_threshold
    if not np.any(valid_scales):
        raise ValueError(
            f"No scales meet the similarity threshold of {similarity_threshold}. "
            f"Mean similarities: {mean_similarities}"
        )

    # Get overall stability score (mean of valid scale similarities)
    valid_sim_matrix = similarity_matrix[np.ix_(valid_scales, valid_scales)]
    stability_score = valid_sim_matrix.mean()
    logger.info(f"Overall stability score: {stability_score:.2f}")

    # Check if overall stability is too low
    if stability_score < overall_threshold:
        raise ValueError(
            f"Overall clustering stability ({stability_score:.2f}) is below threshold "
            f"({overall_threshold}). Results are too unstable across scales."
        )

    # Get valid cluster predictions and compute mode
    valid_preds = all_cluster_preds[valid_scales]
    logger.info(
        f"Using {sum(valid_scales)}/{n_scales} scales: sigma={np.array(sigma_values)[valid_scales]}"
    )

    # Compute mode (most common label at each point)
    combined_cluster_pred, _ = mode(valid_preds, axis=0)
    combined_cluster_pred = combined_cluster_pred.flatten()

    # Compute also average posterior probabilities, entropy and assignment uncertainty
    avg_posterior_probs = np.mean([res["posterior_probs"] for res in results], axis=0)
    avg_entropy = -np.sum(
        avg_posterior_probs * np.log(avg_posterior_probs + 1e-10), axis=1
    )

    # Aggregate results in a dictionary
    aggregated_results = {
        "combined_cluster_pred": combined_cluster_pred,
        "similarity_matrix": similarity_matrix,
        "stability_score": stability_score,
        "valid_scales": np.array(sigma_values)[valid_scales].tolist(),
        "avg_posterior_probs": avg_posterior_probs,
        "avg_entropy": avg_entropy,
    }

    return aggregated_results


# --- Plotting functions ---


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
            cmap="OrRd",
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
            cmap="OrRd",
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

        velocity_arr = np.asarray(velocity)  # Ensure numpy array for range computation
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

        # Compute cluster statistics
        stats = compute_cluster_statistics(
            df_features=df_features,
            cluster_pred=cluster_pred,
            posterior_probs=posterior_probs,
            idata=idata,
            scaler=scaler,
        )
        if not stats:
            raise ValueError("Unable to compute cluster statistics (no clusters?)")

        # Render statistics box
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


# --- DEPRECATED ---


def clusterize_gaussian_mixture_old(
    data_array_scaled: np.ndarray,
    prior_probs: np.ndarray,
    sectors: dict[str, Any],
    output_dir: Path,
    base_name: str,
    df_input: pd.DataFrame,
    scaler: StandardScaler,
    img: np.ndarray | None = None,
    mu_params: dict | None = None,
    sigma_params: dict | None = None,
    feature_weights: np.ndarray | None = None,
    sample_args: dict | None = None,
    apply_mrf_regularization: bool = False,
    mrf_kwargs: dict[str, Any] | None = None,
    second_pass: str = "full",
    second_pass_sample_args: dict | None = None,
    make_plots: bool = True,
    random_seed: int = 8927,
) -> dict:
    """
    Run MCMC-based clustering on preprocessed velocity/features with a Gaussian mixture model and optional MRF regularization.

    Args:
        data_array_scaled (np.ndarray): Scaled feature array for clustering.
        prior_probs (np.ndarray): Prior probabilities array (n_samples, n_clusters).
        sectors (dict): Dictionary of sector names to shapely Polygons.
        output_dir (Path): Directory to save outputs.
        base_name (str): Base name for output files.
        df_input (pd.DataFrame): Original input dataframe (for saving results).
        scaler (StandardScaler): Fitted scaler object.
        img (np.ndarray, optional): Image array for plotting overlays. Default is None.
        mu_params (dict, optional): Parameters for the mean of the mixture components. Default is None.
        sigma_params (dict, optional): Parameters for the standard deviation of the mixture components. Default is None.
        feature_weights (np.ndarray, optional): Optional feature weights for the model. Default is None.
        sample_args (dict, optional): Arguments for PyMC sampling. Default is None.
        apply_mrf_regularization (bool, optional): Whether to apply MRF regularization to spatial priors. Default is False.
        mrf_kwargs (dict, optional): Arguments for MRF regularization. Default is None.
        second_pass (str, optional): Strategy for second sampling pass ("skip", "short", "full"). Default is "full".
        second_pass_sample_args (dict, optional): Arguments for second pass sampling. Default is None.
        make_plots (bool, optional): Whether to generate and save plots. Default is True.
        random_seed (int, optional): Random seed for reproducibility. Default is 8927.

    Returns:
        dict: Dictionary containing results:
            - "idata": ArviZ InferenceData object
            - "scaler": StandardScaler object
            - "convergence_flag": bool
            - "posterior_probs": np.ndarray
            - "cluster_pred": np.ndarray
            - "uncertainty": np.ndarray
    """

    # --- helper: build initvals from idata posterior means (warm-start) ---
    def _initvals_from_idata(idata_in, n_chains):
        mu_mean = idata_in.posterior["mu"].mean(dim=["chain", "draw"]).values
        sigma_mean = idata_in.posterior["sigma"].mean(dim=["chain", "draw"]).values
        # Ensure shapes match the model dims; return a list of per-chain dicts
        init = {"mu": mu_mean, "sigma": sigma_mean}
        return [init for _ in range(n_chains)]

    logger.info(f"Running MCMC clustering for {base_name}...")

    # Default parameters if not provided
    if mu_params is None:
        mu_params = {"mu": 0, "sigma": 1}
    if sigma_params is None:
        sigma_params = {"sigma": 1}
    if sample_args is None:
        sample_args = dict(
            target_accept=0.95,
            draws=2000,
            tune=1000,
            chains=4,
            cores=4,
            random_seed=random_seed,
        )

    posterior_probs = None
    cluster_pred = None
    uncertainty = None

    # Build model
    model = build_marginalized_mixture_model(
        data_array_scaled,
        prior_probs,
        sectors,
        mu_params=mu_params,
        sigma_params=sigma_params,
        feature_weights=feature_weights,
    )

    # Sample model (1st pass)
    idata, convergence_flag = sample_model(model, output_dir, base_name, **sample_args)
    if not convergence_flag:
        idata_summary = az.summary(idata, var_names=["mu", "sigma"])
        logger.info(f"MCMC did not converge. Summary:\n{idata_summary}")

    # Save sampling summary
    save_sampling_summary(convergence_flag, idata, output_dir, f"{base_name}_pass1")

    logger.info("First pass MCMC clustering completed.")

    # --- MRF regularization of priors and optional re-sample ---
    if apply_mrf_regularization:
        if mrf_kwargs is None:
            mrf_kwargs = {
                "n_neighbors": 8,
                "length_scale": 50,
                "beta": 2,
                "n_iter": 5,
            }

        prior_used = prior_probs.copy()
        logger.info("Applying MRF regularization to spatial priors...")
        x_pos = df_input["x"].to_numpy()
        y_pos = df_input["y"].to_numpy()
        prior_mrf, q_mrf = mrf_regularization(
            data_array_scaled, idata, prior_probs, x_pos, y_pos, **mrf_kwargs
        )
        prior_used = prior_mrf

        if make_plots:
            try:
                fig, _ = plot_spatial_priors(df_input, prior_mrf, img=img)
                fig.savefig(
                    output_dir
                    / f"{base_name}_mrf_priors_neig{mrf_kwargs['n_neighbors']}_ls{mrf_kwargs['length_scale']}_beta{mrf_kwargs['beta']}.jpg",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
            except Exception as exc:
                logger.warning(f"Could not plot MRF priors: {exc}")

        # Decide second pass strategy
        if second_pass.lower() == "skip":
            logger.info(
                "Skipping re-sampling after MRF regularization. Using pre-sampled MCMC posteriors."
            )
            posterior_probs = q_mrf
            cluster_pred = np.argmax(posterior_probs, axis=1)
            uncertainty = 1.0 - posterior_probs.max(axis=1)
        else:
            if mrf_regularization:
                logger.info("Re-sampling with MRF-regularized priors...")
                with model:
                    pm.set_data({"prior_w": prior_used})

            sp2_args = dict(**sample_args)
            if second_pass.lower() == "short":
                sp2_args.update(dict(draws=600, tune=400, chains=2, cores=2))
                if second_pass_sample_args:
                    sp2_args.update(second_pass_sample_args)
            elif second_pass_sample_args:
                sp2_args.update(second_pass_sample_args)

            initvals = _initvals_from_idata(idata, sp2_args.get("chains", 2))

            with model:
                idata, convergence_flag = sample_model(
                    model,
                    output_dir,
                    base_name + ("_mrf" if mrf_regularization else ""),
                    initvals=initvals,
                    **sp2_args,
                )
            logger.info("Second pass MCMC clustering completed.")

    # Compute posterior-based assignments
    posterior_probs, cluster_pred, uncertainty = compute_posterior_assignments(
        idata, n_posterior_samples=200
    )

    # Generate plots
    if make_plots:
        fig = plot_velocity_clustering(
            df_features=df_input,
            img=img,
            idata=idata,
            cluster_pred=cluster_pred,
            posterior_probs=posterior_probs,
            scaler=scaler,
        )
        fig.savefig(
            output_dir / f"{base_name}_results.jpg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        az.plot_trace(
            idata, var_names=["mu", "sigma"], axes=axes, compact=True, legend=True
        )
        fig.savefig(output_dir / f"{base_name}_trace_plots.jpg", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        az.plot_forest(
            idata, var_names=["mu", "sigma"], combined=True, ess=True, ax=axes
        )
        fig.savefig(output_dir / f"{base_name}_forest_plot.jpg", dpi=150)
        plt.close(fig)

    # Save main output results as a CSV file
    results_df = df_input.copy()
    results_df["cluster_pred"] = cluster_pred
    results_df["uncertainty"] = uncertainty
    for i in range(posterior_probs.shape[1]):
        results_df[f"posterior_prob_{i}"] = posterior_probs[:, i]
    results_df.to_csv(output_dir / f"{base_name}_mcmc_results.csv", index=False)
    save_sampling_summary(convergence_flag, idata, output_dir, f"{base_name}_pass1")
    logger.info(
        f"Saved MCMC clustering results to {output_dir / f'{base_name}_mcmc_results.csv'}"
    )

    result = {
        "idata": idata,
        "scaler": scaler,
        "convergence_flag": convergence_flag,
        "posterior_probs": posterior_probs,
        "cluster_pred": cluster_pred,
        "uncertainty": uncertainty,
    }

    plt.close("all")
    return result
