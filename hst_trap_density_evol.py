"""
Fit and plot the total trap density from multiple sets of HST ACS images.

First run hst_warm_pixels.py for the datasets.

Parameters
----------
dataset_list : str (opt.)
    The name of the list of image datasets to run. Defaults to "test". See the
    dataset_list_names dictionary for the options, in hst_warm_pixels.py.
"""
from hst_warm_pixels import *


# ========
# Constants
# ========
# Julian date of HST ACS launch
date_acs_launch = 0  ##


# ========
# Utility functions
# ========
def dataset_list_saved_density_evol(list_name):
    """Return the file path for the saved density data for a dataset list."""
    return dataset_root + list_name + "_density_evol.npz"


# ========
# Main functions
# ========
def fit_dataset_total_trap_density(dataset):
    """Load and prepare the stacked-trail data for fit_total_trap_density().

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

    # Concatenate the charge values from all trails
    y_all = np.ravel(stacked_lines.data)
    noise_all = np.ravel(stacked_lines.noises)

    # Duplicate the x arrays for all trails
    length = stacked_lines.lengths[0]
    x_all = np.tile(np.arange(length) + 1, stacked_lines.n_lines)

    # Duplicate the single parameters of each trail for all pixels
    n_e_all = np.repeat(
        np.array([line.mean_flux for line in stacked_lines.lines]), length
    )
    n_bg_all = np.repeat(
        np.array([line.mean_background for line in stacked_lines.lines]), length
    )
    row_all = np.repeat(
        np.array([line.mean_row for line in stacked_lines.lines]), length
    )

    # Run the fitting
    rho_q, rho_q_std = fit_total_trap_density(
        x_all, y_all, noise_all, n_e_all, n_bg_all, row_all
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

    plt.errorbar(
        days,
        densities,
        yerr=density_errors,
        c="k",
        ls="none",
        marker="x",
        capsize=3,
        elinewidth=1,
    )

    plt.xlabel("Days Since ACS Launch")
    plt.ylabel(r"Trap Density per Pixel, $\rho_{\rm q}$")

    save_path = dataset_root + list_name + "_density_evol.png"
    plt.savefig(save_path)
    print("Saved", save_path[-30:])


# ========
# Main
# ========
if __name__ == "__main__":
    # ========
    # Parse arguments
    # ========
    parser = prep_parser()
    args = parser.parse_args()

    if args.dataset_list not in dataset_list_names.keys():
        print("Error: Invalid dataset_list", args.dataset_list)
        print("  Choose from:", list(dataset_list_names.keys()))
        raise ValueError
    dataset_list = dataset_list_names[args.dataset_list]

    # ========
    # Fit and save the total trap densities for each dataset
    # ========
    print("Fit total trap densities...", end=" ", flush=True)
    fit_total_trap_densities(dataset_list, args.dataset_list)

    # ========
    # Plot the trap density evolution
    # ========
    print("Plot trap density evolution...", end=" ", flush=True)
    plot_trap_density_evol(args.dataset_list)
