import logging
import os
import warnings
from typing import List

import numpy as np
import requests
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

from warm_pixels import hst_utilities as ut
from warm_pixels import misc
from warm_pixels.misc import nice_plot
from warm_pixels.misc import plot_hist
from warm_pixels.model.group import QuadrantGroup
from warm_pixels.pixel_lines import PixelLineCollection
from .fit import fit_dataset_total_trap_density, TrapDensities
from .trail_model import trail_model_hst

logger = logging.getLogger(
    __name__
)


def plot_warm_pixels(image, warm_pixels, save_path=None):
    """Plot an image and mark the locations of warm pixels.

    stack_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    image : [[float]]
        The 2D image array.

    warm_pixels : PixelLineCollection
        The set of warm pixel trails.

    save_path : str (opt.)
        The file path for saving the figure. If None, then show the figure.
    """
    # Plot the image and the found warm pixels
    plt.figure()

    im = plt.imshow(
        X=image,
        aspect="equal",
        vmin=0,
        vmax=500
    )
    try:
        plt.scatter(
            warm_pixels.locations[:, 1] + 0.5,
            warm_pixels.locations[:, 0] + 0.5,
            marker=".",
            c="r",
            edgecolor="none",
            s=0.1,
            alpha=0.7,
        )
    except Exception as e:
        logger.exception(e)

    plt.colorbar(im)
    plt.axis("off")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=500)
        plt.close()


def plot_warm_pixel_distributions(quadrants, save_path=None):
    """Plot histograms of the properties of premade warm pixel trails.

    find_dataset_warm_pixels() and find_consistent_warm_pixels() must first be
    run for the dataset.

    Parameters
    ----------
    quadrants : [str]
        The list of quadrants (A, B, C, D) of the images to plot.

    save_path : str (opt.)
        The file path for saving the figure. If None, then show the figure.
    """
    # Tile four histograms
    plt.figure()
    gs = GridSpec(nrows=2, ncols=2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])

    if len(quadrants) > 1:
        colours = misc.A1_c
    else:
        colours = ["k"]

    # Load
    warm_pixels = sum(quadrant.consistent_lines() for quadrant in quadrants)

    # Set bins for all quadrants
    n_row_bins = 15
    n_flux_bins = 30
    n_background_bins = 10
    n_date_bins = 10

    row_min = np.amin(warm_pixels.locations[:, 0])
    row_max = np.amax(warm_pixels.locations[:, 0])
    row_bins = np.linspace(row_min, row_max, n_row_bins + 1)

    flux_min = np.amin(warm_pixels.fluxes)
    flux_max = np.amax(warm_pixels.fluxes)
    flux_bins = np.logspace(np.log10(flux_min), np.log10(flux_max), n_flux_bins + 1)

    background_min = np.amin(warm_pixels.backgrounds)
    background_max = np.amax(warm_pixels.backgrounds)
    background_bins = np.linspace(background_min, background_max, n_background_bins + 1)

    date_min = np.amin(warm_pixels.dates - ut.date_acs_launch)
    date_max = np.amax(warm_pixels.dates - ut.date_acs_launch)
    date_bins = np.linspace(date_min, date_max, n_date_bins + 1)

    # Plot each quadrant separately
    for quadrant, c in zip(
            quadrants,
            colours
    ):
        warm_pixels = quadrant.consistent_lines()
        # Data
        row_hist, row_bin_edges = np.histogram(
            warm_pixels.locations[:, 0], bins=row_bins
        )
        flux_hist, flux_bin_edges = np.histogram(warm_pixels.fluxes, bins=flux_bins)
        background_hist, background_bin_edges = np.histogram(
            warm_pixels.backgrounds, bins=background_bins
        )
        date_hist, date_bin_edges = np.histogram(
            warm_pixels.dates - ut.date_acs_launch, bins=date_bins
        )

        # Plot
        plot_hist(ax1, row_hist, row_bin_edges, c=c)
        plot_hist(ax2, flux_hist, flux_bin_edges, c=c, label=quadrant)
        plot_hist(ax3, background_hist, background_bin_edges, c=c)
        plot_hist(ax4, date_hist, date_bin_edges, c=c)

    ax2.legend(fontsize=12)

    # Axes
    ax1.set_xlabel("Row")
    ax2.set_xlabel(r"Flux (e$^-$)")
    ax3.set_xlabel(r"Background (e$^-$)")
    ax4.set_xlabel("Days Since ACS Launch")
    ax1.set_ylabel("Number of Warm Pixels")
    ax3.set_ylabel("Number of Warm Pixels")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax3.ticklabel_format(useOffset=False, axis="x")
    ax4.ticklabel_format(useOffset=False, axis="x")

    misc.nice_plot(ax1)
    misc.nice_plot(ax2)
    misc.nice_plot(ax3)
    misc.nice_plot(ax4)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
        print("Saved", save_path.stem)


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


def plot_trap_density_evol(
        list_name,
        all_trap_densities: List[TrapDensities],
        do_sunspots=True,
        use_corrected=False,
        do_pdf=False
):
    """Plot the evolution of the total trap density.

    fit_total_trap_densities() must first be run for the dataset list.

    Parameters
    ----------
    list_name : str
        The name of the list of image datasets.

    do_sunspots : bool (opt.)
        Whether or not to also plot the monthly average sunspot number.

    use_corrected : bool (opt.)
        If True, then also plot the results from the corrected images with CTI
        removed.

    do_pdf : bool (opt.)
        If True, then save as a pdf instead of a png.
    """
    # Colours
    if len(all_trap_densities) == 1:
        colours = ["k"]
        colours_cor = ["0.35"]
    else:
        colours = misc.A1_c[: len(all_trap_densities)]
        colours_cor = misc.A1_c[: len(all_trap_densities)]

    # Set date limits
    # npzfile = np.load(ut.dataset_list_saved_density_evol(list_name, quadrant_sets[0], use_corrected=use_corrected))
    # days, densities, errors = [npzfile[var] for var in npzfile.files]
    trap_densities = all_trap_densities[0]
    days = trap_densities.days
    densities = trap_densities.densities
    errors = trap_densities.density_errors

    day_0 = 0
    day_1 = np.amax(days) * 1.02

    # Plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # ========
    # Load and plot sunspot data
    # ========
    if do_sunspots:
        # Load
        # https://wwwbis.sidc.be/silso/datafiles#total monthly mean
        # Year | Month | Decimal year | N sunspots | Std dev | N obs | Provisional?
        sunspot_path = "SN_m_tot_V2.0.txt"
        if not os.path.exists(sunspot_path):
            response = requests.get(
                "https://wwwbis.sidc.be/silso/DATA/SN_ms_tot_V2.0.txt"
            )
            response.raise_for_status()
            with open(sunspot_path, "w+b") as f:
                f.write(response.content)

        sunspot_data = np.genfromtxt(
            "SN_m_tot_V2.0.txt",
            dtype=[("dcml_year", float), ("sunspots", float), ("sunspots_err", float)],
            usecols=(2, 3, 4),
        )
        with warnings.catch_warnings():
            # Ignore astropy.time's "dubious year" warnings
            warnings.simplefilter("ignore")
            sunspot_days = (
                    ut.dec_yr_to_jd(sunspot_data["dcml_year"]) - ut.date_acs_launch
            )

        # Restrict to the relevant dates
        sel_ss = np.where((day_0 < sunspot_days) & (sunspot_days < day_1))[0]
        sunspot_data = sunspot_data[sel_ss]
        sunspot_days = sunspot_days[sel_ss]

        # Plot
        ax2 = ax.twinx()
        ax2.errorbar(
            sunspot_days,
            sunspot_data["sunspots"],
            yerr=sunspot_data["sunspots_err"],
            c="0.8",
            ls="none",
            marker=".",
            capsize=3,
            elinewidth=1,
        )

        # Label on primary axes
        ax.errorbar(
            [],
            [],
            yerr=[],
            c="0.8",
            ls="none",
            marker=".",
            capsize=3,
            elinewidth=1,
            label="Sunspot number",
        )

        # Axes etc
        ax.patch.set_visible(False)
        ax2.patch.set_visible(True)
        ax2.set_zorder(-1)
        ax2.set_ylabel(r"Sunspot Number, Monthly Average")
        ax2.set_ylim(0, None)
        plt.sca(ax)

    # ========
    # Load and plot data
    # ========
    for i_q, trap_densities in enumerate(all_trap_densities):
        days = trap_densities.days
        densities = trap_densities.densities
        errors = trap_densities.density_errors

        label = trap_densities.quadrants_string
        c = colours[i_q]

        # Fit trends
        sel_pre_T_change = np.where(days < ut.day_T_change)[0]
        sel_post_T_change = np.where(days > ut.day_T_change)[0]
        for i_sel, sel in enumerate([sel_pre_T_change, sel_post_T_change]):
            if len(sel) == 0:
                continue

            # Sunspot fit
            if do_sunspots and False:
                # Cumulative sunspot number
                sunspot_cum = np.cumsum(sunspot_data["sunspots"])
                sunspot_cum_err = np.sqrt(np.cumsum(sunspot_data["sunspots"] ** 2))

                # Plot cumulative sunspot number
                if i_sel == 0 and True:  ##
                    ax2.errorbar(
                        days,
                        np.interp(days, sunspot_days, sunspot_cum),
                        yerr=np.interp(days, sunspot_days, sunspot_cum_err),
                        c="0.8",
                        ls="none",
                        marker="o",
                        capsize=3,
                        elinewidth=1,
                    )
                    ax2.set_ylim(0, sunspot_cum[-1] * 1.05)
                    ax2.set_ylabel(r"Cumulative Sunspot Number")
                    plt.sca(ax)
            # Linear fit
            else:
                # Fitting function
                def linear(x, m, c):
                    return m * x + c

                # Fit (around middle t for nicer error plotting)
                day_mid = np.mean(days[sel])
                popt, pcov = curve_fit(
                    linear, days[sel] - day_mid, densities[sel], sigma=errors[sel]
                )
                grad, icpt = popt
                err_grad = np.sqrt(pcov[0, 0])
                err_icpt = np.sqrt(pcov[1, 1])
                if days[sel][-1] > ut.day_T_change:
                    # Extrapolate on to the plot edge
                    days_fit = np.append(days[sel], [day_1])
                    if days[sel][0] < ut.day_side2_fail:
                        # And back to the T change
                        days_fit = np.append([ut.day_T_change], days_fit)
                else:
                    # Extrapolate on to the T change
                    days_fit = np.append(days[sel], [ut.day_T_change])
                    # And back to the plot edge
                    days_fit = np.append([day_0], days_fit)
                fit_densities = linear(days_fit - day_mid, grad, icpt)

                # Plot
                ax.plot(days_fit, fit_densities, c=c, lw=1)
                fit_errors = np.sqrt(
                    err_icpt ** 2 + ((days_fit - day_mid) * err_grad) ** 2
                )
                ax.plot(days_fit, fit_densities + fit_errors, c=c, lw=1, alpha=0.25)
                ax.plot(days_fit, fit_densities - fit_errors, c=c, lw=1, alpha=0.25)

                # Shift for neater function of t
                icpt -= grad * day_mid

                label += str(
                    "\n"
                    + r"$(%.3f \pm %.3f) \!\times\! 10^{-4}\;\, t \,+\, (%.3f \pm %.3f)$"
                    % (grad / 1e-4, err_grad / 1e-4, icpt, err_icpt)
                )

        # Data
        ax.errorbar(
            days,
            densities,
            yerr=errors,
            c=c,
            ls="none",
            marker="x",
            capsize=3,
            elinewidth=1,
            label=label,
        )

        # Corrected images with CTI removed
        if use_corrected:
            c = colours_cor[i_q]

            # Plot negative values separately
            where_pos = np.where(densities > 0)[0]
            where_neg = np.where(densities < 0)[0]

            # Data
            ax.errorbar(
                days[where_pos],
                densities[where_pos],
                yerr=errors[where_pos],
                c=c,
                ls="none",
                marker="x",
                capsize=3,
                elinewidth=1,
                label="After correction",
            )
            ax.scatter(
                days[where_neg],
                abs(densities[where_neg]),
                color=c,
                facecolor="w",
                marker="o",
                zorder=-2,
            )
            ax.errorbar(
                days[where_neg],
                abs(densities[where_neg]),
                yerr=errors[where_neg],
                c=c,
                ls="none",
                capsize=3,
                elinewidth=1,
                zorder=-1,
            )

    # ========
    # HST CTI measurements using Richard's IDL code
    # ========
    # if not True:  ##
    #     # date, density, density_err
    #     data = np.array(
    #         [
    #             [431.303, 0.179387, 0.0682717],  # shortSNe2
    #             [804.024, 0.325217, 0.0512948],  # 05_2004
    #             [1131.27, 0.456763, 0.762311],  # 04_2005
    #             [1519.10, 0.627182, 0.0732714],  # 04_2006
    #             [1599.39, 0.611703, 0.0760443],  # richmassey60490
    #             [1613.18, 0.560601, 0.0496126],  # richmassey61093
    #             [1629.13, 0.632204, 0.0515503],  # richmassey60491
    #             [1655.14, 0.657068, 0.0503882],  # richmassey61092
    #             [2803.10, 1.34501, 0.0720851],  # sm43
    #             [3007.13, 1.45635, 0.0732634],  # 05_2010
    #             [3321.37, 1.65278, 0.0453292],  # 04_2011
    #             [3799.49, 1.89259, 0.0684670],  # huff_spt814b
    #             [4050.26, 2.01314, 0.0802822],  # 04_2013
    #             [4377.37, 2.07898, 0.0479423],  # 02_2014
    #             [4709.00, 2.29900, 0.238915],  # 01_2015
    #             [5058.00, 2.48080, 0.297159],  # 01_2016
    #             [5514.32, 2.69825, 0.0761266],  # 04_2017
    #             [5695.42, 2.58939, 0.0724275],  # 10_2017
    #             [6008.27, 2.84505, 0.351008],  # 08_2018
    #             [6240.09, 3.01478, 0.0649324],  # 04_2019
    #             [6595.34, 3.16847, 0.606145],  # 03_2020
    #             [6852.48, 3.26501, 0.209639],  # 12_2020
    #         ]
    #     )
    #     ax.errorbar(
    #         data[:, 0],
    #         data[:, 1],
    #         yerr=data[:, 2],
    #         ls="none",
    #         marker="+",
    #         capsize=3,
    #         elinewidth=1,
    #     )

    # Axes etc
    ax.set_xlabel("Days Since ACS Launch")
    ax.set_ylabel(r"Total Trap Density per Pixel, $\rho_{\rm q}$")
    ax.set_xlim(day_0, day_1)
    ax.set_ylim(
        min(0, np.amin(densities - 2 * errors)), 1.1 * np.amax(densities + errors)
    )
    ax.xaxis.set_minor_locator(MultipleLocator(200))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Legend
    plt.legend(loc="center left", prop={"size": 14})

    # Mark dates
    ax.axvline(ut.day_T_change, c="k", lw=1)
    ax.axvline(ut.day_side2_fail, c="k", lw=1)
    ax.axvline(ut.day_sm4_repair, c="k", lw=1)
    ax.axvspan(
        ut.day_side2_fail, ut.day_sm4_repair, fc="0.7", ec="none", alpha=0.3, zorder=-1
    )
    for day, text, ha in [
        [ut.day_T_change, "Temperature Change", "right"],
        [ut.day_side2_fail, "Side-2 Failure", "left"],
        [ut.day_sm4_repair, "SM4 Repair", "right"],
    ]:
        if ha == "left":
            x_shift = 1.03
        else:
            x_shift = 0.99
        ax.text(
            day * x_shift,
            0.99,
            text,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            size=16,
            ha=ha,
            va="top",
        )

    # Calendar years
    ax_yr = ax.twiny()
    ax_yr.set_xlabel("Calendar Year")
    ax_yr.set_xlim(day_0, day_1)
    year_ticks = np.arange(2003, ut.jd_to_dec_yr(ut.date_acs_launch + day_1), 1)
    ax_yr.set_xticks(ut.dec_yr_to_jd(year_ticks[1::2]) - ut.date_acs_launch)
    ax_yr.set_xticks(ut.dec_yr_to_jd(year_ticks[::2]) - ut.date_acs_launch, minor=True)
    ax_yr.set_xticklabels(["%d" % year for year in year_ticks[1::2]])

    nice_plot(ax)
    nice_plot(ax_yr)

    save_path = ut.dataset_list_plotted_density_evol(
        list_name, [trap_densities.quadrants_string for trap_densities in all_trap_densities], do_pdf=do_pdf
    )
    plt.savefig(save_path, dpi=200)
    print("Saved", save_path.name)
