import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter
from scipy.stats import mode
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def preproc_features(df):
    """
    Create additional features for clustering glacier displacement data
    """
    df_features = df.copy()

    # Direction angle (in radians and degrees)
    df_features["angle_rad"] = np.arctan2(df["v"], df["u"])
    df_features["angle_deg"] = np.degrees(df_features["angle_rad"])

    # Convert negative angles to positive (0-360 degrees)
    df_features["angle_deg"] = (df_features["angle_deg"] + 360) % 360

    # Directional components (unit vectors)
    df_features["u_unit"] = df["u"] / (df["V"] + 1e-10)  # Avoid division by zero
    df_features["v_unit"] = df["v"] / (df["V"] + 1e-10)

    # Spatial derivatives (approximate gradients)
    # Note: This is a simplified approach - in practice you might want to use proper spatial interpolation
    df_features["spatial_index"] = range(len(df_features))

    # Log magnitude for better clustering of different scales
    df_features["log_magnitude"] = np.log1p(df["V"])

    # Magnitude categories
    magnitude_percentiles = np.percentile(df["V"], [25, 50, 75])
    df_features["magnitude_category"] = pd.cut(
        df["V"],
        bins=[0] + list(magnitude_percentiles) + [np.inf],
        labels=["low", "medium", "high", "very_high"],
    )

    return df_features


def normalize_data(
    df,
    spatial_weight=0.3,
    velocity_weight=0.7,
    spatial_features_names=None,
    velocity_features_names=None,
):
    """
    Custom clustering approach that considers both spatial proximity and velocity similarity

    Parameters:
    - spatial_weight: weight for spatial coordinates (x, y)
    - velocity_weight: weight for velocity features (u, v, magnitude, direction)
    """

    if spatial_features_names is None:
        spatial_features_names = ["x", "y"]
    if velocity_features_names is None:
        velocity_features_names = ["u", "v", "V", "angle_deg"]

    # Prepare features for clustering
    spatial_features = df[spatial_features_names].values
    velocity_features = df[velocity_features_names].values

    # Normalize features separately
    spatial_scaler = StandardScaler()
    velocity_scaler = StandardScaler()

    spatial_normalized = spatial_scaler.fit_transform(spatial_features)
    velocity_normalized = velocity_scaler.fit_transform(velocity_features)

    # Create a DataFrame with normalized features
    normalized_df = pd.DataFrame(
        np.hstack(
            [
                spatial_normalized * spatial_weight,
                velocity_normalized * velocity_weight,
            ]
        ),
        columns=spatial_features_names + velocity_features_names,
    )

    return normalized_df, spatial_scaler, velocity_scaler


def visualize_uv_plt(df, ax=None, **kwargs):
    """
    Visualize the u-v scatter plot with optional background image.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    V = df["V"].values if "V" in df else np.sqrt(df["u"] ** 2 + df["v"] ** 2)
    scatter = ax.scatter(
        df["u"], df["v"], s=1, c=V, alpha=0.6, cmap="viridis", **kwargs
    )
    ax.set_xlabel("u (displacement in x direction)")
    ax.set_ylabel("v (displacement in y direction)")
    ax.set_title("Displacement Vectors (u-v Scatter Plot)")
    plt.colorbar(scatter, ax=ax)
    ax.set_aspect("equal", adjustable="box")


def visualize_pca(df, columns_to_extract=None, normalize=False):
    "visualize the enhanced features using PCA for dimensionality reduction"
    from sklearn.decomposition import PCA

    # Prepare data for PCA
    if columns_to_extract is None:
        columns_to_extract = ["x", "y", "u", "v", "V", "angle_deg"]

    # Ensure all required columns are present in the DataFrame
    missing_columns = set(columns_to_extract) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(df[columns_to_extract])
    else:
        data = df[columns_to_extract].values

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    df_reduced = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_reduced["PC1"], df_reduced["PC2"], s=1, alpha=0.6)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Reduced Features Scatter Plot")
    ax.set_aspect("equal", adjustable="box")


def plot_gmm_clusters(df, labels, var_names=None, img=None, figsize=(15, 10)):
    """
    Visualize clustering results on the glacier displacement field.
    Simplified version inspired by plot_dic_vectors function.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Get unique labels and create colors
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)

    # Use a colormap for consistent colors
    if n_clusters > 0:
        cmap = plt.get_cmap("Set3")
        colors = cmap(np.linspace(0, 1, max(n_clusters, 3)))
        # Handle noise points with red color
        color_map = {}
        cluster_idx = 0
        for label in unique_labels:
            if label == -1:
                color_map[label] = "red"
            else:
                color_map[label] = colors[cluster_idx % len(colors)]
                cluster_idx += 1
    else:
        color_map = {-1: "red"}

    # Plot 1: Spatial distribution of clusters
    ax1 = axes[0, 0]
    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            cluster_name = "Noise" if label == -1 else f"Cluster {label}"
            ax1.scatter(
                df.loc[mask, "x"],
                df.loc[mask, "y"],
                c=[color_map[label]],
                s=2,
                alpha=0.7,
                label=cluster_name,
            )

    ax1.set_title("Spatial Distribution")
    ax1.invert_yaxis()  # Match image coordinates
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.set_aspect("equal")
    ax1.grid(False)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot 2: Displacement vectors colored by velocity magnitude (like plot_dic_vectors)
    ax2 = axes[0, 1]
    if img is not None:
        ax2.imshow(img, alpha=0.7)
    q = ax2.quiver(
        df["x"],
        df["y"],
        df["u"],
        df["v"],
        df["V"],  # Color by velocity magnitude
        scale=None,
        scale_units="xy",
        angles="xy",
        cmap="viridis",
        width=0.003,
        headwidth=2.5,
        alpha=0.8,
    )
    cbar = fig.colorbar(q, ax=ax2)
    cbar.set_label("Displacement Magnitude (pixels)")
    ax2.set_title("Displacement Vectors")
    ax2.grid(False)
    ax2.set_aspect("equal")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot 3: 2D scatter plot of two variables (e.g., angle vs magnitude)
    ax3 = axes[1, 0]
    if var_names is None:
        var_names = ["angle_deg", "V"]
    if len(var_names) != 2:
        raise ValueError(
            "var_names must contain exactly two variable names for the 2D scatter plot."
        )
    if not all(var in df.columns for var in var_names):
        raise ValueError(
            f"One or both of the specified variables {var_names} do not exist in the DataFrame."
        )
    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            cluster_name = "Noise" if label == -1 else f"Cluster {label}"
            ax3.scatter(
                df.loc[mask, var_names[0]],
                df.loc[mask, var_names[1]],
                c=[color_map[label]],
                s=15,
                alpha=0.6,
                label=cluster_name,
            )
    ax3.set_title("2D Scatter Plot")
    ax3.set_xlabel(var_names[0])
    ax3.set_ylabel(var_names[1])
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Cluster statistics
    ax4 = axes[1, 1]
    cluster_stats = []
    for label in unique_labels:
        mask = labels == label
        cluster_name = "Noise" if label == -1 else f"Cluster {label}"

        stats = {
            "Cluster": cluster_name,
            "Count": np.sum(mask),
            "Mean_Magnitude": df.loc[mask, "V"].mean(),
            "Mean_Direction": df.loc[mask, "angle_deg"].mean(),
            "Std_Magnitude": df.loc[mask, "V"].std(),
        }
        cluster_stats.append(stats)
    stats_df = pd.DataFrame(cluster_stats)

    # Bar plot of cluster sizes
    bars = ax4.bar(
        range(len(stats_df)),
        stats_df["Count"],
        color=[color_map[label] for label in unique_labels],
    )

    ax4.set_title("Cluster Sizes")
    ax4.set_xlabel("Cluster")
    ax4.set_ylabel("Number of Points")
    ax4.set_xticks(range(len(stats_df)))
    ax4.set_xticklabels(stats_df["Cluster"], rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, axes, stats_df


def plot_gmm_log_likelihood_contours(
    df,
    gmm,
    scaler,
    variables_names,
    pair=None,
    n_grid=80,
    ax=None,
    cmap="viridis",
    alpha=0.5,
):
    """
    Plot GMM negative log-likelihood contours for any pair of variables in n-dimensional space.
    - df: DataFrame with all features
    - gmm: fitted GaussianMixture
    - scaler: fitted StandardScaler
    - variables_names: list of all variable names used for GMM
    - pair: tuple/list of two variable names to plot (default: first two)
    - n_grid: grid resolution
    - ax: matplotlib axis (optional)
    - cmap: colormap for contours
    - alpha: contour transparency
    """
    import matplotlib.colors as mcolors

    if pair is None:
        pair = variables_names[:2]
    assert len(pair) == 2, "pair must be a tuple/list of two variable names"
    idx1, idx2 = [variables_names.index(v) for v in pair]

    # Prepare grid in the selected 2D space
    data_scaled = scaler.transform(df[variables_names])
    xlim = (
        np.floor(np.percentile(data_scaled[:, idx1], 1)),
        np.ceil(np.percentile(data_scaled[:, idx1], 99)),
    )
    ylim = (
        np.floor(np.percentile(data_scaled[:, idx2], 1)),
        np.ceil(np.percentile(data_scaled[:, idx2], 99)),
    )
    x = np.linspace(*xlim, n_grid)
    y = np.linspace(*ylim, n_grid)
    X, Y = np.meshgrid(x, y)
    grid = np.zeros((n_grid * n_grid, data_scaled.shape[1]))
    # Fill grid with mean values for all other variables
    grid[:] = np.mean(data_scaled, axis=0)
    grid[:, idx1] = X.ravel()
    grid[:, idx2] = Y.ravel()
    # Compute negative log-likelihood
    Z = -gmm.score_samples(grid)
    Z = Z.reshape(X.shape)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    levels = np.linspace(np.percentile(Z, 5), np.percentile(Z, 99), 10)
    CS = ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        cmap=cmap,
        alpha=alpha,
        linewidths=1.5,
        norm=mcolors.Normalize(vmin=levels[0], vmax=levels[-1]),
    )
    CB = fig.colorbar(CS, ax=ax, shrink=0.8, extend="both")
    CB.set_label("Negative log-likelihood")
    # Scatter the data points
    ax.scatter(
        data_scaled[:, idx1], data_scaled[:, idx2], s=2, c="k", alpha=0.3, label="Data"
    )
    ax.set_xlabel(f"{pair[0]} (scaled)")
    ax.set_ylabel(f"{pair[1]} (scaled)")
    ax.set_title(f"GMM Negative log-likelihood: {pair[0]} vs {pair[1]}")
    ax.axis("tight")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return ax


def remove_small_clusters(labels, min_size=30):
    """Relabel clusters with fewer than min_size points as noise (-1)."""
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    small_clusters = unique[counts < min_size]
    for c in small_clusters:
        labels[labels == c] = -1
    return labels


def spatial_smooth_labels(df, labels, window_size=3):
    """
    Apply a spatial mode filter to smooth cluster labels.
    Assumes df has 'x', 'y' columns and labels is a 1D array.
    """
    # Create a grid for fast lookup
    x, y = df["x"].astype(int), df["y"].astype(int)
    grid = np.full((y.max() + 1, x.max() + 1), -1, dtype=int)
    grid[y, x] = labels

    def mode_filter(values):
        vals = values[values != -1]
        return mode(vals, keepdims=False)[0] if len(vals) > 0 else -1

    smoothed_grid = generic_filter(
        grid, mode_filter, size=window_size, mode="constant", cval=-1
    )

    # Map back to original points
    smoothed_labels = smoothed_grid[y, x]
    return smoothed_labels


def merge_similar_clusters(df, labels, threshold=10):
    """Merge clusters whose centroids are closer than a threshold."""
    unique_labels = np.unique(labels[labels != -1])
    centroids = np.array(
        [
            df[labels == l][["x", "y", "V", "angle_rad"]].mean().values
            for l in unique_labels
        ]
    )
    dists = pairwise_distances(centroids)
    np.fill_diagonal(dists, np.inf)
    merged = labels.copy()
    for i, l1 in enumerate(unique_labels):
        for j, l2 in enumerate(unique_labels):
            if i < j and dists[i, j] < threshold:
                merged[merged == l2] = l1
    return merged
