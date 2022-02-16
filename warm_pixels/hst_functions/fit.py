"""Primary and plotting functions for hst_warm_pixels.py"""

import lmfit
import numpy as np

from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import PixelLineCollection
from . import trail_model


def fit_total_trap_density(x_all, y_all, noise_all, n_e_all, n_bg_all, row_all, date, use_arctic=False):
    """Fit the total trap density for a trail or a concatenated set of trails.

    Other than the x, y, and noise values, which should cover all pixels in the
    trail or set of trails, the parameters must be either a single value or an
    array of the same length. So if the data are a concatenated set of multiple
    trails, e.g. x_all = [1, 2, ..., n, 1, 2, ..., n, 1, ...], then e.g. row_all
    should be [row_1, row_1, ..., row_1, row_2, row_2, ... row_2, row_3, ...] to
    set the correct values for all pixels in each trail. The date is taken as a
    single value, which only affects the results by being before vs after the
    change of trap model.

    Parameters
    ----------
    x_all : [float]
        The pixel positions away from the trailed pixel.

    y_all : [float]
        The charge values.

    noise_all : float or [float]
        The charge noise error value.

    n_e_all : float or [float]
        The number of electrons in the trailed pixel's charge cloud (e-).

    n_bg_all : float or [float]
        The background number of electrons (e-).

    row_all : float or [float]
        Distance in pixels of the trailed pixel from the readout register.

    date : float
        The Julian date of the images, used to set the trap model.

    Returns
    -------
    rho_q : float
        The best-fit total number density of traps per pixel.

    rho_q_std : float
        The standard error on the total trap density.
    """

    # Initialise the fitting model
    if use_arctic:
        model = lmfit.models.Model(
            func=trail_model.trail_model_hst_arctic, independent_vars=["x", "n_e", "n_bg", "row", "date"]
        )
    else:
        model = lmfit.models.Model(
            func=trail_model.trail_model_hst, independent_vars=["x", "n_e", "n_bg", "row", "date"]
        )
    params = model.make_params()

    # Initialise the fit
    params["rho_q"].value = 1.0
    params["rho_q"].min = 0.0

    # Weight using the noise
    weights = 1 / noise_all ** 2

    # Run the fitting
    result = model.fit(
        data=y_all,
        params=params,
        weights=weights,
        x=x_all,
        n_e=n_e_all,
        n_bg=n_bg_all,
        row=row_all,
        date=date,
    )
    # print(result.fit_report())  ##

    return result.params.get("rho_q").value, result.params.get("rho_q").stderr, result.eval()


def fit_dataset_total_trap_density(
        dataset, quadrants, use_arctic=False,
        row_bins=None, flux_bins=None, background_bins=None
):
    """Load, prep, and pass the stacked-trail data to fit_total_trap_density().

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrants : [str]
        The list of quadrants (A, B, C, D) of the images to load, combined
        together if more than one provided.

    use_corrected : bool (opt.)
        If True, then use the corrected images with CTI removed instead.

    Returns
    -------
    rho_q : float
        The best-fit total number density of traps per pixel.

    rho_q_std : float
        The standard error on the total trap density.
    """
    # Load
    stacked_lines = PixelLineCollection()
    stacked_lines.load(dataset.saved_stacked_lines(quadrants))
    npzfile = np.load(dataset.saved_stacked_info(quadrants))
    n_row_bins, n_flux_bins, n_date_bins, n_background_bins = [
        (len(npzfile[var]) - 1) for var in npzfile.files
    ]

    # Decide which bin to fit
    if row_bins is None:
        row_bins = range(n_row_bins)
    if flux_bins is None:
        flux_bins = range(n_flux_bins)
    if background_bins is None:
        background_bins = range(n_background_bins)

    # Compile the data from all stacked lines
    n_lines_used = 0
    y_all = np.array([])
    noise_all = np.array([])
    n_e_each = np.array([])
    n_bg_each = np.array([])
    row_each = np.array([])

    # ========
    # Concatenate each stacked trail
    # ========
    for i_row in row_bins:
        for i_flux in flux_bins:
            for i_background in background_bins:
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

                    #
                    # Compile data into easy form to fit
                    #
                    y_all = np.append(y_all, line.model_trail)
                    noise_all = np.append(noise_all, line.model_trail_noise)
                    n_e_each = np.append(n_e_each, line.model_flux)
                    n_bg_each = np.append(n_bg_each, line.model_background)
                    row_each = np.append(row_each, line.mean_row)
                    n_lines_used += 1

    if n_lines_used == 0:
        return None, None, np.zeros(ut.trail_length)

    # Duplicate the x arrays for all trails
    x_all = np.tile(np.arange(ut.trail_length) + 1, n_lines_used)

    # Duplicate the single parameters of each trail for all pixels
    n_e_all = np.repeat(n_e_each, ut.trail_length)
    n_bg_all = np.repeat(n_bg_each, ut.trail_length)
    row_all = np.repeat(row_each, ut.trail_length)

    # Run the fitting
    rho_q, rho_q_std, y_fit = fit_total_trap_density(
        x_all, y_all, noise_all, n_e_all, n_bg_all, row_all, dataset.date, use_arctic=use_arctic
    )

    return rho_q, rho_q_std, y_fit


def fit_total_trap_densities(dataset_list, list_name, quadrants, use_corrected=False):
    """Call fit_dataset_total_trap_density() for each dataset and compile and
    save the results.

    Parameters
    ----------
    dataset_list : [str]
        The list of image datasets to run.

    list_name : str
        The name of the list of image datasets.

    quadrants : [str]
        The list of quadrants (A, B, C, D) of the images to load, combined
        together if more than one provided.

    use_corrected : bool (opt.)
        If True, then use the corrected images with CTI removed instead.

    Saves
    -----
    days : [float]
    densities : [float]
    density_errors : [float]
        The date (days since launch), total trap density, and standard error on
        the density for each dataset in the list, saved to
        dataset_list_saved_density_evol().
    """
    # Initialise arrays
    days = []
    densities = []
    density_errors = []

    # Analyse each dataset
    for i_dataset, dataset in enumerate(dataset_list):
        print(
            "\rFit total trap densities (%s)... "
            '"%s" (%d of %d)'
            % ("".join(quadrants), dataset.name, i_dataset + 1, len(dataset_list)),
            end="            ",
            flush=True,
        )

        # Fit the density
        rho_q, rho_q_std, y_fit = fit_dataset_total_trap_density(
            dataset, quadrants, use_corrected
        )

        # Skip bad fits
        if rho_q is None or rho_q_std is None:
            print("# error")
            continue

        # Append the data
        days.append(dataset.date - ut.date_acs_launch)
        densities.append(rho_q)
        density_errors.append(rho_q_std)
    print("\rFit total trap densities (%s)... " % "".join(quadrants))

    if len(days) == 0:
        raise AssertionError(
            "No successful fits"
        )

    # Sort
    sort = np.argsort(days)
    days = np.array(days)[sort]
    densities = np.array(densities)[sort]
    density_errors = np.array(density_errors)[sort]

    # Save
    np.savez(
        ut.dataset_list_saved_density_evol(list_name, quadrants, use_corrected),
        days,
        densities,
        density_errors,
    )
