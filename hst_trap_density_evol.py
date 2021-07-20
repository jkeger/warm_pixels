"""
Fit and plot the total trap density from multiple sets of HST ACS images.

First run hst_warm_pixels.py to prepare the datasets.

Parameters
----------
dataset_list : str (opt.)
    The name of the list of image datasets to run. Defaults to "test". See the
    dataset_lists dictionary for the options, in hst_warm_pixels.py.

--mdate_old_* : str (opt.)
    A "year/month/day" requirement to remake files saved/modified before this
    date. Defaults to only check whether a file already exists. Alternatively,
    set "1" to force remaking or "0" to force not.
    --mdate_old_all, -a
        Overrides all others.
    --mdate_old_ttd, -t
        Total trap density.
    --mdate_old_pde, -e
        Plot density evolution.
"""
from hst_warm_pixels import *

from scipy.optimize import curve_fit


# ========
# Utility functions
# ========
def prep_parser_extra(parser):
    parser.add_argument(
        "-t",
        "--mdate_old_ttd",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for total trap density.",
    )
    parser.add_argument(
        "-e",
        "--mdate_old_pde",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for plot density evolution.",
    )

    return parser


def dataset_list_saved_density_evol(list_name):
    """Return the file path for the saved density data for a dataset list."""
    return dataset_root + "density_evol_%s.npz" % list_name


def dataset_list_plotted_density_evol(list_name):
    """Return the file path for the saved density plot for a dataset list."""
    return path + "/density_evol_%s.png" % list_name


# ========
# Main functions
# ========
def fit_total_trap_densities(dataset_list, list_name):
    """Call fit_dataset_total_trap_density() for each dataset and compile and
    save the results.

    Parameters
    ----------
    dataset_list : [str]
        The list of image datasets to run.

    Saves
    -----
    days : [float]
    densities : [float]
    density_errors : [float]
        The date (days since launch), total trap density, and standard error on
        the density for each dataset in the list, saved to
        dataset_list_saved_density_evol(list_name).
    """
    # Initialise arrays
    days = []
    densities = []
    density_errors = []

    # Analyse each dataset
    for i_dataset, dataset in enumerate(dataset_list):
        print(
            "\rFit total trap densities... "
            '"%s" (%d of %d)' % (dataset.name, i_dataset + 1, len(dataset_list)),
            end="            ",
            flush=True,
        )

        # Fit the density
        rho_q, rho_q_std = fit_dataset_total_trap_density(dataset)

        # Skip bad fits
        if rho_q is None or rho_q_std is None:
            print("# error")
            continue

        # Append the data
        days.append(dataset.date - date_acs_launch)
        densities.append(rho_q)
        density_errors.append(rho_q_std)
    print("\rFit total trap densities... ")

    # Save
    np.savez(
        dataset_list_saved_density_evol(list_name), days, densities, density_errors
    )


# ========
# Plotting functions
# ========
def plot_trap_density_evol(list_name):
    """Plot the evolution of the total trap density.

    Parameters
    ----------
    list_name : str
        The name of the list of image datasets.
    """
    # Load
    npzfile = np.load(dataset_list_saved_density_evol(list_name))
    days, densities, errors = [npzfile[var] for var in npzfile.files]

    # ========
    # Plot
    # ========
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Data
    ax.errorbar(
        days,
        densities,
        yerr=errors,
        c="k",
        ls="none",
        marker="x",
        capsize=3,
        elinewidth=1,
    )

    # Linear fits
    def linear(x, m, c):
        return m * x + c

    sel_pre_T_change = np.where(days < day_T_change)[0]
    sel_post_T_change = np.where(days > day_T_change)[0]
    for sel in [sel_pre_T_change, sel_post_T_change]:
        popt, pcov = curve_fit(linear, days[sel], densities[sel], sigma=errors[sel])
        grad, icpt = popt
        err_grad = np.sqrt(pcov[0, 0])
        err_icpt = np.sqrt(pcov[1, 1])
        ax.plot(
            days[sel],
            linear(days[sel], grad, icpt),
            label=r"$(%.2g \pm %.2g) \!\times\! 10^{-4}\; t + (%.2g \pm %.2g)$"
            % (grad / 1e-4, err_grad / 1e-4, icpt, err_icpt),
        )
    plt.legend(loc="lower right")

    # Axes etc
    ax.set_xlabel("Days Since ACS Launch")
    ax.set_ylabel(r"Total Trap Density per Pixel, $\rho_{\rm q}$")
    day_0 = 0
    day_1 = np.amax(days) * 1.02
    ax.set_xlim(day_0, day_1)
    ax.set_ylim(
        min(0, np.amin(densities - 2 * errors)), 1.1 * np.amax(densities + errors)
    )

    # Mark dates
    ax.axvline(day_T_change, c="k", lw=1)
    ax.axvline(day_side2_fail, c="k", lw=1)
    ax.axvline(day_repair, c="k", lw=1)
    ax.axvspan(day_side2_fail, day_repair, fc="0.7", ec="none", alpha=0.3, zorder=-1)
    for day, text, ha in [
        [day_T_change, "Temperature Change", "right"],
        [day_side2_fail, "Side-2 Failure", "left"],
        [day_repair, "SM4 Repair", "right"],
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
    ax2 = ax.twiny()
    ax2.set_xlabel("Calendar Year")
    ax2.set_xlim(day_0, day_1)
    year_ticks = np.arange(2003, jd_to_dec_yr(date_acs_launch + day_1), 1)
    ax2.set_xticks(dec_yr_to_jd(year_ticks[1::2]) - date_acs_launch)
    ax2.set_xticks(dec_yr_to_jd(year_ticks[::2]) - date_acs_launch, minor=True)
    ax2.set_xticklabels(["%d" % year for year in year_ticks[1::2]])

    my_nice_plot(ax)
    my_nice_plot(ax2)

    save_path = dataset_list_plotted_density_evol(list_name)
    plt.savefig(save_path)
    print("Saved", save_path[-40:])


# ========
# Main
# ========
if __name__ == "__main__":
    # ========
    # Parse arguments
    # ========
    parser = prep_parser()
    parser = prep_parser_extra(parser)
    args = parser.parse_args()

    list_name = args.dataset_list
    if list_name not in dataset_lists.keys():
        print("Error: Invalid dataset_list", list_name)
        print("  Choose from:", list(dataset_lists.keys()))
        raise ValueError
    dataset_list = dataset_lists[list_name]

    if args.mdate_old_all is not None:
        args.mdate_old_ttd = args.mdate_old_all
        args.mdate_old_pde = args.mdate_old_all

    # Fit and save the total trap densities for each dataset
    if need_to_make_file(
        dataset_list_saved_density_evol(list_name), date_old=args.mdate_old_ttd
    ):
        print("Fit total trap densities...", end=" ", flush=True)
        fit_total_trap_densities(dataset_list, list_name)

    # Plot the trap density evolution
    if need_to_make_file(
        dataset_list_plotted_density_evol(list_name), date_old=args.mdate_old_pde
    ):
        print("Plot trap density evolution...", end=" ", flush=True)
        plot_trap_density_evol(list_name)
