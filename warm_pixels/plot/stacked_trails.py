import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from warm_pixels import hst_utilities as ut
from warm_pixels import misc
from warm_pixels.hst_functions.fit import fit_dataset_total_trap_density
from warm_pixels.hst_functions.trail_model import trail_model_hst
from warm_pixels.model.group import QuadrantGroup
from warm_pixels.pixel_lines import PixelLineCollection

logger = logging.getLogger(
    __name__
)


def plot_stacked_trails(group: QuadrantGroup, use_corrected=False, save_path=None):
    """Plot a tiled set of stacked trails.

    stack_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrants : [str]
        The list of quadrants (A, B, C, D) of the images to load, combined
        together if more than one provided.

    use_corrected : bool (opt.)
        If True, then use the corrected images with CTI removed instead.

    save_path : str (opt.)
        The file path for saving the figure. If None, then show the figure.
    """
    stacked_lines = group.stacked_lines()

    n_row_bins = stacked_lines.n_row_bins
    n_flux_bins = stacked_lines.n_flux_bins
    n_background_bins = stacked_lines.n_background_bins

    # Plot the stacked trails
    plt.figure(figsize=(25, 12))
    gs = GridSpec(n_row_bins, n_flux_bins)
    axes = [
        [plt.subplot(gs[i_row, i_flux]) for i_flux in range(n_flux_bins)]
        for i_row in range(n_row_bins)
    ]
    gs.update(wspace=0, hspace=0)

    # Don't plot the warm pixel itself
    pixels = np.arange(1, ut.trail_length + 1)
    sel_non_zero = np.where(stacked_lines.data[:, -ut.trail_length:] != 0)
    # Set y limits
    if use_corrected:
        # For symlog scale
        # Assume ymin < 0
        y_min = 0.1  # 4 * np.amin(stacked_lines.data[:, -ut.trail_length :][sel_non_zero])
        y_max = 4 * np.amax(stacked_lines.data[:, -ut.trail_length:][sel_non_zero])
        log10_y_min = np.ceil(np.log10(abs(y_min)))
        log10_y_max = np.floor(np.log10(y_max))
        y_min = min(y_min, -10 ** (log10_y_min + 0.6))
        y_max = max(y_max, 10 ** (log10_y_max + 0.6))
        y_ticks = np.append(
            -10 ** np.arange(log10_y_min, -0.1, -1),
            10 ** np.arange(0, log10_y_max + 0.1, 1),
        )
    else:
        # For log scale
        y_min = np.partition(
            abs(np.ravel(stacked_lines.data[:, -ut.trail_length:][sel_non_zero])), 2
        )[1]
        y_min = 0.1
        y_max = 4 * np.amax(stacked_lines.data[:, -ut.trail_length:][sel_non_zero])
        log10_y_min = np.ceil(np.log10(y_min))
        log10_y_max = np.floor(np.log10(y_max))
        y_min = min(y_min, 10 ** (log10_y_min - 0.4))
        y_max = max(y_max, 10 ** (log10_y_max + 0.4))
        y_ticks = 10 ** np.arange(log10_y_min, log10_y_max + 0.1, 1)
    if n_background_bins == 1:
        colours = ["k"]
    else:
        colours = plt.cm.jet(np.linspace(0.05, 0.95, n_background_bins))

    # Label size
    fontsize = 14

    # Fit the total trap density to the full dataset
    print("Performing global fit")
    rho_q_set, rho_q_std_set, y_fit = fit_dataset_total_trap_density(
        group, use_arctic=False
    )
    print(rho_q_set, rho_q_std_set, "exponentials")

    # Fit the total trap density to the full dataset using arCTIc
    # print("Performing global fit")
    # rho_q_set, rho_q_std_set, y_fit = fit_dataset_total_trap_density(
    #     dataset, quadrants, use_corrected=use_corrected, use_arctic=True
    # )
    # print(rho_q_set, rho_q_std_set, "ArCTIc")

    # Fit the total trap density to the full dataset using arCTIc and MCMC
    # print("Performing global fit using arCTIc")
    # result = fit_warm_pixels_with_arctic(
    #    dataset, quadrants, use_corrected=use_corrected
    # )

    # Fit to each trail individually, and plot as we go along
    print("Performing individual fits:")
    for i_row in range(n_row_bins):
        for i_flux in range(n_flux_bins):
            # Furthest row bin at the top
            ax = axes[n_row_bins - 1 - i_row][i_flux]

            # Plot each background bin's stack
            for i_background, c in enumerate(colours):

                bin_index = PixelLineCollection.stacked_bin_index(
                    i_row=i_row,
                    i_flux=i_flux,
                    n_flux_bins=n_flux_bins,
                    i_background=i_background,
                    n_background_bins=n_background_bins,
                )

                line = (stacked_lines.lines[bin_index])
                # Skip empty and single-entry bins
                if line.n_stacked <= 1:
                    continue

                # Don't plot the warm pixel itself
                trail = line.model_trail  # + line.model_background
                noise = line.model_trail_noise  # + line.model_background

                # Check for negative values
                where_pos = np.where(trail > 0)[0]
                where_neg = np.where(trail < 0)[0]

                # ========
                # Plot data
                # ========
                if use_corrected:
                    # Plot positives and negatives together for symlog scale
                    ax.errorbar(
                        pixels, trail, yerr=noise, color=c, capsize=2, alpha=0.7
                    )
                else:
                    # Plot positives and negatives separately for log scale
                    ax.errorbar(
                        pixels[where_pos],
                        trail[where_pos],
                        yerr=noise[where_pos],
                        color=c,
                        capsize=2,
                        alpha=0.7,
                    )
                    ax.scatter(
                        pixels[where_neg],
                        abs(trail[where_neg]),
                        color=c,
                        facecolor="w",
                        marker="o",
                        alpha=0.7,
                        zorder=-1,
                    )
                    ax.errorbar(
                        pixels[where_neg],
                        abs(trail[where_neg]),
                        yerr=noise[where_neg],
                        color=c,
                        fmt=",",
                        alpha=0.7,
                        zorder=-2,
                    )

                # ========
                # Plot fitted trail
                # ========
                # Fit the total trap density to this single stacked trail (dotted line, which has swapped since Jacob's version)
                rho_q_indiv, rho_q_std_indiv, y_fit_indiv = fit_dataset_total_trap_density(
                    group, use_arctic=True,
                    row_bins=[i_row], flux_bins=[i_flux], background_bins=[i_background]
                )
                ax.plot(pixels, y_fit_indiv, color=c, ls=misc.ls_dot, alpha=0.7)

                # Also reconstruct then plot the simultaneous fit to all trails (dashed line)
                model_trail = trail_model_hst(
                    x=pixels,
                    rho_q=rho_q_set,
                    n_e=line.mean_flux,
                    n_bg=line.mean_background,
                    row=line.mean_row,
                    date=group.dataset.date,
                )
                ax.plot(pixels, model_trail, color=c, ls=misc.ls_dash, alpha=0.7)

                # Annotate
                if i_background == 0:
                    text = "$%d$" % line.n_stacked
                else:
                    text = "\n" * i_background + "$%d$" % line.n_stacked
                ax.text(
                    0.97,
                    0.96,
                    text,
                    transform=ax.transAxes,
                    size=fontsize,
                    ha="right",
                    va="top",
                )

            ax.set_xlim(0.5, ut.trail_length + 0.5)
            ax.set_xticks(np.arange(2, ut.trail_length + 0.1, 2))
            ax.set_xticks(np.arange(1, ut.trail_length + 0.1, 2), minor=True)
            if use_corrected:
                ax.set_yscale("symlog", linthreshy=1, linscaley=0.5)
                ax.axhline(0, lw=0.5, c="0.7", zorder=-99)
            else:
                ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)

            # Axis labels
            if i_row != 0:
                ax.set_xticklabels([])
            elif i_flux in [2, n_flux_bins - 3]:
                ax.set_xlabel("Pixel")
            if i_flux != 0:
                ax.set_yticklabels([])
            elif i_row in [1, n_row_bins - 2]:
                ax.set_ylabel("Number of electrons (e$^-$)")

            # Bin edge labels
            if i_flux == n_flux_bins - 1:
                if i_row == 0:
                    ax.text(
                        1.02,
                        0.5,
                        "Row:",
                        transform=ax.transAxes,
                        rotation=90,
                        ha="left",
                        va="center",
                    )
                if i_row < n_row_bins - 1:
                    ax.text(
                        1.02,
                        1.0,
                        "%d" % stacked_lines.row_bins[i_row + 1],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="left",
                        va="center",
                    )
            if i_row == n_row_bins - 1:
                if i_flux == 0:
                    ax.text(
                        0.5,
                        1.01,
                        r"Hot pixel (e$^-$):",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                    )
                flux_max = stacked_lines.flux_bins[i_flux + 1]
                pow10 = np.floor(np.log10(flux_max))
                text = r"$%.1f \!\times\! 10^{%d}$" % (flux_max / 10 ** pow10, pow10)
                ax.text(
                    1.0, 1.01, text, transform=ax.transAxes, ha="center", va="bottom"
                )
            if i_row == int(n_row_bins / 2) and i_flux == n_flux_bins - 1:
                text = "Background:  "
                for i_background in range(n_background_bins):
                    text += "%.0f$-$%.0f" % (
                        stacked_lines.background_bins[i_background],
                        stacked_lines.background_bins[i_background + 1],
                    )
                    if i_background < n_background_bins - 1:
                        text += ",  "
                ax.text(
                    1.25,
                    0.5,
                    text,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="left",
                    va="center",
                )

            # Total trap density
            if i_row == n_row_bins - 1 and i_flux == n_flux_bins - 1:
                if rho_q_set is None or rho_q_std_set is None:
                    text = "fit error"
                else:
                    text = r"$\rho_{\rm q} = %.3f \pm %.3f$" % (
                        rho_q_set,
                        rho_q_std_set,
                    )
                ax.text(
                    0.03,
                    0.03,
                    text,
                    transform=ax.transAxes,
                    size=fontsize,
                    ha="left",
                    va="bottom",
                )

            # Tidy
            if i_row == 0 and i_flux == 0:
                misc.set_large_ticks(ax)
            elif i_row == 0:
                misc.set_large_ticks(ax, do_y=False)
            elif i_flux == 0:
                misc.set_large_ticks(ax, do_x=False)
            misc.set_font_size(ax)

    plt.tight_layout()

    if save_path is None:
        plt.show()
    elif save_path == "None":
        return
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print("Saved", save_path.name)
