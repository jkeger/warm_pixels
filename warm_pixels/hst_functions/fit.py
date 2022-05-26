"""Primary and plotting functions for hst_warm_pixels.py"""
from typing import Tuple

import lmfit
import numpy as np

from warm_pixels import hst_utilities as ut
from warm_pixels.model.group import QuadrantGroup
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

    eval
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
        group: QuadrantGroup,
        use_arctic=False,
        row_bins=None,
        flux_bins=None,
        background_bins=None
):
    """Load, prep, and pass the stacked-trail data to fit_total_trap_density().

    Parameters
    ----------
    group
        A group of quadrants and a dataset for which consistent stacked lines are computed.

    Returns
    -------
    rho_q : float
        The best-fit total number density of traps per pixel.

    rho_q_std : float
        The standard error on the total trap density.

    y_fit
    """
    # Load
    stacked_lines = group.stacked_lines()

    # Decide which bin to fit
    if row_bins is None:
        row_bins = range(stacked_lines.n_row_bins)
    if flux_bins is None:
        flux_bins = range(stacked_lines.n_flux_bins)
    if background_bins is None:
        background_bins = range(stacked_lines.n_background_bins)

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
                line = stacked_lines.stacked_line_for_indices(
                    row_index=i_row,
                    flux_index=i_flux,
                    background_index=i_background,
                    date_index=0,
                )

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
        x_all, y_all, noise_all, n_e_all, n_bg_all, row_all, group.dataset.date, use_arctic=use_arctic
    )

    return rho_q, rho_q_std, y_fit


def fit_total_trap_densities(groups: Tuple[QuadrantGroup]):
    """Call fit_dataset_total_trap_density() for each dataset and compile and
    save the results.

    Parameters
    ----------
    groups
        Groups of quadrants from datasets over time
    """
    # Initialise arrays
    days = []
    densities = []
    density_errors = []

    # Analyse each dataset
    for i_dataset, group in enumerate(groups):
        print(
            "\rFit total trap densities... "
            '"%s" (%d of %d)'
            % (group.dataset.name, i_dataset + 1, len(groups)),
            end="            ",
            flush=True,
        )

        # Fit the density
        rho_q, rho_q_std, y_fit = fit_dataset_total_trap_density(group)

        # Skip bad fits
        if rho_q is None or rho_q_std is None:
            print("# error")
            continue

        # Append the data
        days.append(group.dataset.date - ut.date_acs_launch)
        densities.append(rho_q)
        density_errors.append(rho_q_std)
    print("\rFit total trap densities")

    if len(days) == 0:
        raise AssertionError(
            "No successful fits"
        )

    # Sort
    sort = np.argsort(days)
    days = np.array(days)[sort]
    densities = np.array(densities)[sort]
    density_errors = np.array(density_errors)[sort]

    return TrapDensities(
        quadrants_string=str(groups[0]),
        days=days,
        densities=densities,
        density_errors=density_errors,
    )


class TrapDensities:
    def __init__(
            self,
            quadrants_string,
            days,
            densities,
            density_errors,
    ):
        self.quadrants_string = quadrants_string
        self.days = days
        self.densities = densities
        self.density_errors = density_errors
