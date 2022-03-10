import logging
import os
import warnings
from typing import List

import numpy as np
import requests
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

from warm_pixels import hst_utilities as ut
from warm_pixels import misc
from warm_pixels.hst_functions.fit import TrapDensities
from warm_pixels.misc import nice_plot

logger = logging.getLogger(
    __name__
)


def plot_trap_density_evol(
        save_path,
        all_trap_densities: List[TrapDensities],
        use_corrected=False,
):
    """Plot the evolution of the total trap density.

    fit_total_trap_densities() must first be run for the dataset list.

    Parameters
    ----------
    save_path
        The path at which the plot should be saved
    all_trap_densities
        A list of objects each describing how trap densities vary over time for
        a group of quadrants
    use_corrected : bool (opt.)
        If True, then also plot the results from the corrected images with CTI
        removed.
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

            # # Sunspot fit
            # if do_sunspots and False:
            #     # Cumulative sunspot number
            #     sunspot_cum = np.cumsum(sunspot_data["sunspots"])
            #     sunspot_cum_err = np.sqrt(np.cumsum(sunspot_data["sunspots"] ** 2))
            #
            #     # Plot cumulative sunspot number
            #     if i_sel == 0 and True:  ##
            #         ax2.errorbar(
            #             days,
            #             np.interp(days, sunspot_days, sunspot_cum),
            #             yerr=np.interp(days, sunspot_days, sunspot_cum_err),
            #             c="0.8",
            #             ls="none",
            #             marker="o",
            #             capsize=3,
            #             elinewidth=1,
            #         )
            #         ax2.set_ylim(0, sunspot_cum[-1] * 1.05)
            #         ax2.set_ylabel(r"Cumulative Sunspot Number")
            #         plt.sca(ax)
            # # Linear fit
            #
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

    plt.savefig(save_path, dpi=200)
