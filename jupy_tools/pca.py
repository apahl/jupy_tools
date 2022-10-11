"""
Helpers for PCA analysis.
MOst of the tools here were copied
from https://github.com/mpimp-comas/2021_grigalunas_burhop_zinken/blob/master/utils.py
and written by Jose-Manuel Gally.
"""

from typing import List
import itertools

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns


def get_pca_feature_contrib(pca_model: PCA, features: list) -> pd.DataFrame:
    """Get the feature contribution to each Principal Component.
    Parameters:
    ===========
    model: The PCA object
    descriptors_list: The list of feature names that were used for the PCA.
    Returns:
    ========
    A DataFrame with the feature contribution.
    """
    # associate features and pc feature contribution
    ds = []
    for pc in pca_model.components_:
        ds.append(
            {k: np.abs(v) for k, v in zip(features, pc)}
        )  # absolute value of contributions because only the magnitude of the contribution is of interest
    df_feature_contrib = (
        pd.DataFrame(ds, index=[f"PC{i+1}_feature_contrib" for i in range(3)])
        .T.reset_index()
        .rename({"index": "Feature"}, axis=1)
    )

    # compute PC ranks
    for c in df_feature_contrib.columns:
        if not c.endswith("_feature_contrib"):
            continue
        df_feature_contrib = df_feature_contrib.sort_values(
            c, ascending=False
        ).reset_index(drop=True)
        df_feature_contrib[f"{c.split('_')[0]}_rank"] = df_feature_contrib.index + 1

    # add PC-wise ratios
    pattern = "_feature_contrib"
    for c in df_feature_contrib:
        if c.endswith(pattern):
            tot = df_feature_contrib[c].sum()
            df_feature_contrib = df_feature_contrib.sort_values(c, ascending=False)
            df_feature_contrib[
                f"{c.replace(pattern, '')}_feature_contrib_cum_ratio"
            ] = (df_feature_contrib[c].cumsum() / tot)

    return df_feature_contrib.sort_values("Feature").reset_index(drop=True)


def plot_pca_cum_feature_contrib_3pc(
    df_pca_feature_contrib: pd.DataFrame,
) -> plt.Figure:
    """Plot the cumulated feature contribution to each Principal Component Combination
    individually (up to 3 different PCs).
    Parameters:
    ===========
    df_pca_feature_contrib: a DataFrame with the feature contributions
    Returns:
    ========
    A multi barchart, with a subplot for each Principal Component Combination.
    """
    # set up a multiplot for 3 subplots on a same row
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 12), sharey=True)

    # configure plot
    fig.suptitle("Cumulated Feature Contribution to Principal Components", fontsize=30)
    # sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    # sns.set_context("paper", font_scale=2)
    fig.subplots_adjust(hspace=0.2, wspace=0.2, top=0.8)
    x_label = "Features"
    y_label = "Cumulated % of Feature Contribution"

    # iterate over combinations of PCs (PC1 and PC2, PC1 and PC3 and PC2 and PC3)
    pcs = sorted(
        list(set([c.split("_")[0] for c in df_pca_feature_contrib.columns if "_" in c]))
    )
    for i, pc in enumerate(pcs):
        # create the a subplot
        axes[i].set_title(pc, fontsize=30)
        col_y = f"{pc}_feature_contrib_cum_ratio"
        sns.barplot(
            ax=axes[i],
            x="Feature",
            y=col_y,
            data=df_pca_feature_contrib.sort_values(col_y),
            color="gray",
            zorder=2,
        )
        # add x label and ticks for first plot only
        if i == 0:
            # y label
            axes[i].set_ylabel(y_label, fontsize=25, labelpad=20)
            # y ticklabels
            yticklabels = [f"{x:,.0%}" for x in axes[i].get_yticks()]
            axes[i].set_yticklabels(yticklabels)
        else:
            axes[i].set_ylabel("")

        # x label
        axes[i].set_xlabel(x_label, fontsize=20, labelpad=20)
        axes[i].tick_params(axis="x", rotation=90)
        axes[i].axhline(0.5, ls="--", color="red", zorder=1)

    fig.subplots_adjust(bottom=1.0, top=2.5)
    plt.tight_layout()
    return fig


def plot_pca_loadings_3pc(
    pca_model: PCA, pca: np.ndarray, features: List[str], color_dots: str = "white"
):
    """Plot the principal component loadings. By default, only the arrows
    and the feature labels are plotted. If the color_dots parameter is modified,
    then a biplot is generated instead.
    Parameters:
    ===========
    pca_model: the PCA model
    pca: the PCA data
    descriptors_list: the list of feature names that were used for the PCA.
    Returns:
    ========
    A PCA loadings plot or a PCA biplot.
    """

    # data initialization
    pcs = [f"PC{i}" for i in range(1, pca.shape[1] + 1)]
    pc_pairs = list(itertools.combinations(pcs, 2))
    scores = pca
    coefficients = np.transpose(pca_model.components_)

    # plot initialization
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(24, 7), sharex=True, sharey=True
    )
    axes = axes.ravel()
    # add main title
    fig.suptitle("Principal Component Loading", fontsize=30)
    # sns.set_style("whitegrid", {"axes.edgecolor": "0.2"})
    # sns.set_context("paper", font_scale=2)
    fig.subplots_adjust(hspace=0.2, wspace=0.2, top=0.8)

    # generate one subplot at the time
    for i, ax in enumerate(axes):
        pc_pair = pc_pairs[i]
        # determine what columns to retrieve from the pca matrix
        idx_x = int(pc_pair[0].replace("PC", "")) - 1
        idx_y = int(pc_pair[1].replace("PC", "")) - 1
        # retrieve values
        scores_x = scores[:, idx_x]
        scores_y = scores[:, idx_y]
        coefficients_curr = coefficients[:, [idx_x, idx_y]]
        # zoom in Principal Components space
        n = coefficients_curr.shape[0]
        scale_x = 1.0 / (scores_x.max() - scores_x.min())
        scale_y = 1.0 / (scores_y.max() - scores_y.min())
        # plot all data points as white just to get the appropriate coordinates
        ax.scatter(
            x=scores_x * scale_x, y=scores_y * scale_y, s=5, color="white"
        )  # we interest ourselves in the loadings, this is not a biplot
        # add eigenvectors as annotated arrows
        for j in range(n):
            ax.arrow(
                0,
                0,
                coefficients_curr[j, 0],
                coefficients_curr[j, 1],
                color="gray",
                alpha=0.8,
                head_width=0.015,
            )
            ax.text(
                coefficients_curr[j, 0],
                coefficients_curr[j, 1],
                features[j],
                color="red",
                ha="center",
                va="center",
                fontsize=11,
            )

        # finish subplots
        ax.set_title(f"{' and '.join(pc_pair)}", fontsize=20)
        ax.set_xlim([-0.8, 0.8])
        ax.set_xlabel(pc_pair[0], fontsize=20, labelpad=20)
        ax.set_ylabel(pc_pair[1], fontsize=20, labelpad=20)

    return fig
