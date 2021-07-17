"""
Fit and plot the total trap density from multiple sets of HST ACS images.

First run hst_warm_pixels.py to prepare the datasets.

Parameters
----------
dataset_list : str (opt.)
    The name of the list of image datasets to run. Defaults to "test". See the
    dataset_list_names dictionary for the options, in hst_warm_pixels.py.

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
def fit_dataset_total_trap_density(dataset):
    """Load, prep, and pass the stacked-trail data to fit_total_trap_density().

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    Returns
    -------
    rho_q : float
        The best-fit total number density of traps per pixel.

    rho_q_std : float
        The standard error on the total trap density.
    """
    # Load
    stacked_lines = PixelLineCollection()
    stacked_lines.load(dataset.saved_stacked_lines)
    npzfile = np.load(dataset.saved_stacked_info)
    row_bins, flux_bins, date_bins, background_bins = [
        npzfile[var] for var in npzfile.files
    ]
    n_row_bins = len(row_bins) - 1
    n_flux_bins = len(flux_bins) - 1
    n_date_bins = len(date_bins) - 1
    n_background_bins = len(background_bins) - 1

    # Compile the data from all stacked lines
    n_lines_used = 0
    y_all = np.array([])
    noise_all = np.array([])
    n_e_each = np.array([])
    n_bg_each = np.array([])
    row_each = np.array([])

    # Loop over each stacked trail
    # Skip the lowest-row and lowest-flux bins
    for i_row in range(1, n_row_bins):
        for i_flux in range(1, n_flux_bins):
            for i_background in range(n_background_bins):
                bin_index = PixelLineCollection.stacked_bin_index(
                    i_row=i_row,
                    n_row_bins=n_row_bins,
                    i_flux=i_flux,
                    n_flux_bins=n_flux_bins,
                    i_background=i_background,
                    n_background_bins=n_background_bins,
                )

                line = stacked_lines.lines[bin_index]

                if line.n_stacked >= 3:
                    # Concatenate the data
                    y_all = np.append(y_all, line.data[-trail_length:])
                    noise_all = np.append(noise_all, line.noise[-trail_length:])
                    n_e_each = np.append(n_e_each, line.mean_flux)
                    n_bg_each = np.append(n_bg_each, line.mean_background)
                    row_each = np.append(row_each, line.mean_row)
                    n_lines_used += 1

    # Duplicate the x arrays for all trails
    x_all = np.tile(np.arange(trail_length) + 1, n_lines_used)

    # Duplicate the single parameters of each trail for all pixels
    n_e_all = np.repeat(n_e_each, trail_length)
    n_bg_all = np.repeat(n_bg_each, trail_length)
    row_all = np.repeat(row_each, trail_length)

    # Run the fitting
    rho_q, rho_q_std = fit_total_trap_density(
        x_all, y_all, noise_all, n_e_all, n_bg_all, row_all, dataset.date
    )

    return rho_q, rho_q_std


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
        # Date
        days.append(dataset.date - date_acs_launch)

        # Trap density
        rho_q, rho_q_std = fit_dataset_total_trap_density(dataset)

        densities.append(rho_q)
        density_errors.append(rho_q_std)

    # Save
    np.savez(
        dataset_list_saved_density_evol(list_name), days, densities, density_errors
    )
    print("")


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
    days, densities, density_errors = [npzfile[var] for var in npzfile.files]

    # ========
    # Plot
    # ========
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    ax.errorbar(
        days,
        densities,
        yerr=density_errors,
        c="k",
        ls="none",
        marker="x",
        capsize=3,
        elinewidth=1,
    )

    # Axes etc
    ax.set_xlabel("Days Since ACS Launch")
    ax.set_ylabel(r"Total Trap Density per Pixel, $\rho_{\rm q}$")
    day_0 = 0
    day_1 = days[-1] * 1.01
    ax.set_xlim(day_0, day_1)

    # Mark dates
    ax.axvline(date_T_change - date_acs_launch, c="k")
    ax.axvline(date_side2_fail - date_acs_launch, c="k")
    ax.axvline(date_repair - date_acs_launch, c="k")
    ax.axvspan(
        date_side2_fail - date_acs_launch,
        date_repair - date_acs_launch,
        fc="0.7",
        ec="none",
        alpha=0.3,
        zorder=-1,
    )
    for date, text, ha in [
        [date_T_change, "Temperature Change", "right"],
        [date_side2_fail, "Side-2 Failure", "left"],
        [date_repair, "SM4 Repair", "right"],
    ]:
        if ha == "left":
            x_shift = 1.02
        else:
            x_shift = 0.99
        ax.text(
            (date - date_acs_launch) * x_shift,
            0.995,
            text,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            size=14,
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
    print("Saved", save_path[-36:])


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
    if list_name not in dataset_list_names.keys():
        print("Error: Invalid dataset_list", list_name)
        print("  Choose from:", list(dataset_list_names.keys()))
        raise ValueError
    dataset_list = dataset_list_names[list_name]

    if args.mdate_old_all is not None:
        args.mdate_old_ttd = args.mdate_old_all
        args.mdate_old_pde = args.mdate_old_all

    # ========
    # Fit and save the total trap densities for each dataset
    # ========
    if need_to_make_file(
        dataset_list_saved_density_evol(list_name), date_old=args.mdate_old_ttd
    ):
        print("Fit total trap densities...", end=" ", flush=True)
        fit_total_trap_densities(dataset_list, list_name)

    # ========
    # Plot the trap density evolution
    # ========
    if need_to_make_file(
        dataset_list_plotted_density_evol(list_name), date_old=args.mdate_old_pde
    ):
        print("Plot trap density evolution...", end=" ", flush=True)
        plot_trap_density_evol(list_name)
