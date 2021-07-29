"""Primary and plotting functions for hst_warm_pixels.py"""

import numpy as np
import os
import sys
import lmfit
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

from pixel_lines import PixelLineCollection
from warm_pixels import find_warm_pixels

import hst_utilities as ut
from misc import *  # Plotting defaults etc

sys.path.append(os.path.join(ut.path, "../PyAutoArray/"))
import autoarray as aa


# ========
# Main functions
# ========
def find_dataset_warm_pixels(dataset, quadrant):
    """Find the possible warm pixels in all images in a dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrant : str (opt.)
        The quadrant (A, B, C, D) of the image to load.

    Saves
    -----
    warm_pixels : PixelLineCollection
        The set of warm pixel trails, saved to dataset.saved_lines().
    """
    # Initialise the collection of warm pixel trails
    warm_pixels = PixelLineCollection()
    print("")

    # Find the warm pixels in each image
    for i_image in range(dataset.n_images):
        image_path = dataset.image_paths[i_image]
        image_name = dataset.image_names[i_image]
        print(
            "    %s_%s (%d of %d): "
            % (image_name, quadrant, i_image + 1, dataset.n_images),
            end="",
            flush=True,
        )

        # Load the image
        image = aa.acs.ImageACS.from_fits(
            file_path=image_path,
            quadrant_letter=quadrant,
            bias_subtract_via_bias_file=True,
            bias_subtract_via_prescan=True,
        ).native

        date = 2400000.5 + image.header.modified_julian_date

        image_name_q = image_name + "_%s" % quadrant

        # Find the warm pixel trails
        new_warm_pixels = find_warm_pixels(
            image=image, trail_length=ut.trail_length, origin=image_name_q, date=date
        )
        print("Found %d possible warm pixels " % len(new_warm_pixels))

        # Plot
        plot_warm_pixels(
            image,
            PixelLineCollection(new_warm_pixels),
            save_path=dataset.path + image_name_q,
        )

        # Add them to the collection
        warm_pixels.append(new_warm_pixels)

    # Save
    warm_pixels.save(dataset.saved_lines(quadrant))


def find_consistent_warm_pixels(dataset, quadrant):
    """Find the consistent warm pixels in a dataset.

    find_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrant : str (opt.)
        The quadrant (A, B, C, D) of the image to load.

    Saves
    -----
    warm_pixels : PixelLineCollection
        The set of consistent warm pixel trails, saved to
        dataset.saved_consistent_lines().
    """
    # Load
    warm_pixels = PixelLineCollection()
    warm_pixels.load(dataset.saved_lines(quadrant))

    # Find the warm pixels present in at least 2/3 of the images
    consistent_lines = warm_pixels.find_consistent_lines(fraction_present=2 / 3)
    print(
        "Found %d consistents of %d possibles"
        % (len(consistent_lines), warm_pixels.n_lines)
    )

    # Extract the consistent warm pixels
    warm_pixels.lines = warm_pixels.lines[consistent_lines]

    # Save
    warm_pixels.save(dataset.saved_consistent_lines(quadrant))


def stack_dataset_warm_pixels(dataset, quadrants):
    """Stack a set of premade warm pixel trails into bins.

    find_dataset_warm_pixels() and find_consistent_warm_pixels() must first be
    run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrants : [str]
        The list of quadrants (A, B, C, D) of the images to load, combined
        together if more than one provided.

    Saves
    -----
    stacked_lines : PixelLineCollection
        The set of stacked pixel trails, saved to dataset.saved_stacked_lines().
    """
    # Load
    warm_pixels = PixelLineCollection()
    # Append data from each quadrant
    for quadrant in quadrants:
        warm_pixels.load(dataset.saved_consistent_lines(quadrant))

    # Subtract preceeding pixels in each line before stacking
    for i in range(warm_pixels.n_lines):
        warm_pixels.lines[i].data[ut.trail_length + 1 :] -= warm_pixels.lines[i].data[
            : ut.trail_length
        ][::-1]

    # Stack the lines in bins by distance from readout and total flux
    n_row_bins = 5
    n_flux_bins = 10
    n_background_bins = 1
    (
        stacked_lines,
        row_bins,
        flux_bins,
        date_bins,
        background_bins,
    ) = warm_pixels.generate_stacked_lines_from_bins(
        n_row_bins=n_row_bins,
        n_flux_bins=n_flux_bins,
        n_background_bins=n_background_bins,
        return_bin_info=True,
    )
    print("Stacked lines in %d bins" % (n_row_bins * n_flux_bins * n_background_bins))

    # Save
    stacked_lines.save(dataset.saved_stacked_lines(quadrants))
    np.savez(
        dataset.saved_stacked_info(quadrants),
        row_bins,
        flux_bins,
        date_bins,
        background_bins,
    )


def trail_model(x, rho_q, n_e, n_bg, row, alpha, w, A, B, C, tau_a, tau_b, tau_c):
    """Calculate the model shape of a CTI trail.

    Parameters
    ----------
    x : [float]
        The pixel positions away from the trailed pixel.

    rho_q : float
        The total trap number density per pixel.

    n_e : float
        The number of electrons in the trailed pixel's charge cloud (e-).

    n_bg : float
        The background number of electrons (e-).

    row : float
        The distance in pixels of the trailed pixel from the readout register.

    alpha : float
        The CCD well fill power.

    w : float
        The CCD full well depth (e-).

    A, B, C : float
        The relative density of each trap species.

    tau_a, tau_b, tau_c : float
        The release timescale of each trap species (s).

    Returns
    -------
    trail : [float]
        The model charge values at each pixel in the trail (e-).
    """
    return (
        rho_q
        * ((n_e / w) ** alpha - (n_bg / w) ** alpha)
        * row
        * (
            A * np.exp((1 - x) / tau_a) * (1 - np.exp(-1 / tau_a))
            + B * np.exp((1 - x) / tau_b) * (1 - np.exp(-1 / tau_b))
            + C * np.exp((1 - x) / tau_c) * (1 - np.exp(-1 / tau_c))
        )
    )


def trail_model_hst(x, rho_q, n_e, n_bg, row, date):
    """Wrapper for trail_model() for HST ACS.

    Parameters (where different to trail_model())
    ----------
    date : float
        The Julian date of the images, used to set the trap model.

    Returns
    -------
    trail : [float]
        The model charge values at each pixel in the trail (e-).
    """
    # CCD
    alpha = 0.478
    w = 84700.0
    # Trap species
    A = 0.17
    B = 0.45
    C = 0.38
    # Trap lifetimes before or after the temperature change
    if date < ut.date_T_change:
        tau_a = 0.48
        tau_b = 4.86
        tau_c = 20.6
    else:
        tau_a = 0.74
        tau_b = 7.70
        tau_c = 37.0

    return trail_model(x, rho_q, n_e, n_bg, row, alpha, w, A, B, C, tau_a, tau_b, tau_c)


def fit_total_trap_density(x_all, y_all, noise_all, n_e_all, n_bg_all, row_all, date):
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
        The Background number of electrons (e-).

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
    model = lmfit.models.Model(
        func=trail_model_hst,
        independent_vars=["x", "n_e", "n_bg", "row", "date"],
        nan_policy="propagate",  ## not "omit"? If any needed at all?
    )
    params = model.make_params()

    # Initialise the fit
    params["rho_q"].value = 0.1
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
    # print(result.fit_report())

    return result.params.get("rho_q").value, result.params.get("rho_q").stderr


def fit_dataset_total_trap_density(dataset, quadrants):
    """Load, prep, and pass the stacked-trail data to fit_total_trap_density().

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrants : [str]
        The list of quadrants (A, B, C, D) of the images to load, combined
        together if more than one provided.

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

    # ========
    # Concatenate each stacked trail
    # ========
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
                    y_all = np.append(y_all, line.data[-ut.trail_length :])
                    noise_all = np.append(noise_all, line.noise[-ut.trail_length :])
                    n_e_each = np.append(n_e_each, line.mean_flux)
                    n_bg_each = np.append(n_bg_each, line.mean_background)
                    row_each = np.append(row_each, line.mean_row)
                    n_lines_used += 1

    # Duplicate the x arrays for all trails
    x_all = np.tile(np.arange(ut.trail_length) + 1, n_lines_used)

    # Duplicate the single parameters of each trail for all pixels
    n_e_all = np.repeat(n_e_each, ut.trail_length)
    n_bg_all = np.repeat(n_bg_each, ut.trail_length)
    row_all = np.repeat(row_each, ut.trail_length)

    # Run the fitting
    rho_q, rho_q_std = fit_total_trap_density(
        x_all, y_all, noise_all, n_e_all, n_bg_all, row_all, dataset.date
    )

    return rho_q, rho_q_std


def fit_total_trap_densities(dataset_list, list_name, quadrants):
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
        rho_q, rho_q_std = fit_dataset_total_trap_density(dataset, quadrants)

        # Skip bad fits
        if rho_q is None or rho_q_std is None:
            print("# error")
            continue

        # Append the data
        days.append(dataset.date - ut.date_acs_launch)
        densities.append(rho_q)
        density_errors.append(rho_q_std)
    print("\rFit total trap densities (%s)... " % "".join(quadrants))

    # Sort
    sort = np.argsort(days)
    days = np.array(days)[sort]
    densities = np.array(densities)[sort]
    density_errors = np.array(density_errors)[sort]

    # Save
    np.savez(
        ut.dataset_list_saved_density_evol(list_name, quadrants),
        days,
        densities,
        density_errors,
    )


# ========
# Plotting functions
# ========
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

    im = plt.imshow(X=image, aspect="equal", vmin=0, vmax=500)
    plt.scatter(
        warm_pixels.locations[:, 1],
        warm_pixels.locations[:, 0],
        marker=".",
        c="r",
        edgecolor="none",
        s=0.1,
        alpha=0.7,
    )

    plt.colorbar(im)
    plt.axis("off")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=500)
        plt.close()


def plot_warm_pixel_distributions(dataset, quadrants, save_path=None):
    """Plot histograms of the properties of premade warm pixel trails.

    find_dataset_warm_pixels() and find_consistent_warm_pixels() must first be
    run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

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
        colours = A1_c[: len(quadrants)]
    else:
        colours = ["k"]

    # Load
    warm_pixels = PixelLineCollection()
    # Append data from each quadrant
    for quadrant in quadrants:
        warm_pixels.load(dataset.saved_consistent_lines(quadrant))

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
    for quadrant, c in zip(quadrants, colours):
        # Load only this quadrant
        warm_pixels = PixelLineCollection()
        warm_pixels.load(dataset.saved_consistent_lines(quadrant))

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

    nice_plot(ax1)
    nice_plot(ax2)
    nice_plot(ax3)
    nice_plot(ax4)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
        print("Saved", save_path[-36:])


def plot_stacked_trails(dataset, quadrants, save_path=None):
    """Plot a tiled set of stacked trails.

    stack_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrants : [str]
        The list of quadrants (A, B, C, D) of the images to load, combined
        together if more than one provided.

    save_path : str (opt.)
        The file path for saving the figure. If None, then show the figure.
    """
    # Load
    stacked_lines = PixelLineCollection()
    stacked_lines.load(dataset.saved_stacked_lines(quadrants))
    npzfile = np.load(dataset.saved_stacked_info(quadrants))
    row_bins, flux_bins, date_bins, background_bins = [
        npzfile[var] for var in npzfile.files
    ]
    n_row_bins = len(row_bins) - 1
    n_flux_bins = len(flux_bins) - 1
    n_date_bins = len(date_bins) - 1
    n_background_bins = len(background_bins) - 1

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
    sel_non_zero = np.where(stacked_lines.data[:, -ut.trail_length :] != 0)
    y_min = np.partition(
        abs(np.ravel(stacked_lines.data[:, -ut.trail_length :][sel_non_zero])), 2
    )[1]
    y_max = 4 * np.amax(stacked_lines.data[:, -ut.trail_length :][sel_non_zero])
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
    rho_q_set, rho_q_std_set = fit_dataset_total_trap_density(dataset, quadrants)

    # Plot each stack
    for i_row in range(n_row_bins):
        for i_flux in range(n_flux_bins):
            # Furthest row bin at the top
            ax = axes[n_row_bins - 1 - i_row][i_flux]

            for i_background, c in enumerate(colours):
                bin_index = PixelLineCollection.stacked_bin_index(
                    i_row=i_row,
                    n_row_bins=n_row_bins,
                    i_flux=i_flux,
                    n_flux_bins=n_flux_bins,
                    i_background=i_background,
                    n_background_bins=n_background_bins,
                )

                line = stacked_lines.lines[bin_index]

                # Skip empty and single-entry bins
                if line.n_stacked <= 1:
                    continue

                # Don't plot the warm pixel itself
                trail = line.data[-ut.trail_length :]
                noise = line.noise[-ut.trail_length :]

                # Check for negative values
                where_pos = np.where(trail > 0)[0]
                where_neg = np.where(trail < 0)[0]

                # ========
                # Plot data
                # ========
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
                    facecolor="none",
                    marker="o",
                    alpha=0.7,
                )

                # ========
                # Plot fitted trail
                # ========
                # Fit the total trap density to this single stacked trail
                rho_q, rho_q_std = fit_total_trap_density(
                    x_all=pixels,
                    y_all=trail,
                    noise_all=noise,
                    n_e_all=line.mean_flux,
                    n_bg_all=line.mean_background,
                    row_all=line.mean_row,
                    date=dataset.date,
                )
                model_pixels = np.linspace(1, ut.trail_length, 20)
                model_trail = trail_model_hst(
                    x=model_pixels,
                    rho_q=rho_q,
                    n_e=line.mean_flux,
                    n_bg=line.mean_background,
                    row=line.mean_row,
                    date=dataset.date,
                )
                ax.plot(model_pixels, model_trail, color=c, ls="--", alpha=0.7)
                # Also plot the full-set fit
                model_trail = trail_model_hst(
                    x=model_pixels,
                    rho_q=rho_q_set,
                    n_e=line.mean_flux,
                    n_bg=line.mean_background,
                    row=line.mean_row,
                    date=dataset.date,
                )
                ax.plot(model_pixels, model_trail, color=c, ls=":", alpha=0.7)

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
                ax.set_ylabel("Charge (e$^-$)")

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
                        "%d" % row_bins[i_row + 1],
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
                        r"Flux (e$^-$):",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                    )
                flux_max = flux_bins[i_flux + 1]
                pow10 = np.floor(np.log10(flux_max))
                text = r"$%.1f \!\times\! 10^{%d}$" % (flux_max / 10 ** pow10, pow10)
                ax.text(
                    1.0, 1.01, text, transform=ax.transAxes, ha="center", va="bottom"
                )
            if i_row == int(n_row_bins / 2) and i_flux == n_flux_bins - 1:
                text = "Background:  "
                for i_background in range(n_background_bins):
                    text += "%.0f$-$%.0f" % (
                        background_bins[i_background],
                        background_bins[i_background + 1],
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
                    text = r"$\rho_{\rm q} = %.2g \pm %.2g$" % (
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
                set_large_ticks(ax)
            elif i_row == 0:
                set_large_ticks(ax, do_y=False)
            elif i_flux == 0:
                set_large_ticks(ax, do_x=False)
            set_font_size(ax)

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print("Saved", save_path[-40:])


def plot_trap_density_evol(list_name, quadrant_sets, do_sunspots=True):
    """Plot the evolution of the total trap density.

    fit_total_trap_densities() must first be run for the dataset list.

    Parameters
    ----------
    list_name : str
        The name of the list of image datasets.

    quadrant_sets : [[str]]
        The list of quadrants (A, B, C, D) of the images to load, optionally in
        subsets to be combined together.

        e.g. [["A", "B"]] to combine vs [["A"], ["B"]] to keep separate.

    do_sunspots : bool (opt.)
        Whether or not to also plot the monthly average sunspot number.
    """
    # Linear fits
    def linear(x, m, c):
        return m * x + c

    # Colours
    if len(quadrant_sets) == 1:
        colours = ["k"]
    else:
        colours = A1_c[: len(quadrant_sets)]

    # Plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # ========
    # Load and plot data
    # ========
    for i_q, quadrants in enumerate(quadrant_sets):
        # Load
        npzfile = np.load(ut.dataset_list_saved_density_evol(list_name, quadrants))
        days, densities, errors = [npzfile[var] for var in npzfile.files]

        day_0 = 0
        day_1 = np.amax(days) * 1.02

        label = "".join(quadrants)
        c = colours[i_q]

        # Fit trends
        sel_pre_T_change = np.where(days < ut.day_T_change)[0]
        sel_post_T_change = np.where(days > ut.day_T_change)[0]
        for sel in [sel_pre_T_change, sel_post_T_change]:
            if len(sel) == 0:
                continue

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
            fit_errors = np.sqrt(err_icpt ** 2 + ((days_fit - day_mid) * err_grad) ** 2)
            ax.plot(days_fit, fit_densities + fit_errors, c=c, lw=1, alpha=0.25)
            ax.plot(days_fit, fit_densities - fit_errors, c=c, lw=1, alpha=0.25)

            # Shift for neater function of t
            icpt -= grad * day_mid

            label += str(
                "\n"
                + r"$(%.2g \pm %.2g) \!\times\! 10^{-4}\;\, t \,+\, (%.2g \pm %.2g)$"
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

    # ========
    # Sunspots
    # ========
    if do_sunspots:
        ax2 = ax.twinx()

        # Load
        # https://wwwbis.sidc.be/silso/datafiles#total monthly mean
        # Year | Month | Decimal year | N sunspots | Std dev | N obs | Provisional?
        sunspot_data = np.genfromtxt(
            "SN_m_tot_V2.0.txt",
            dtype=[("dcml_year", float), ("sunspots", float), ("sunspots_err", float)],
            usecols=(2, 3, 4),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sunspot_days = (
                ut.dec_yr_to_jd(sunspot_data["dcml_year"]) - ut.date_acs_launch
            )
        sel_ss = np.where((day_0 < sunspot_days) & (sunspot_days < day_1))[0]

        # Plot
        ax2.errorbar(
            sunspot_days[sel_ss],
            sunspot_data["sunspots"][sel_ss],
            yerr=sunspot_data["sunspots_err"][sel_ss],
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
    plt.legend(loc="lower right", prop={"size": 14})

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

    save_path = ut.dataset_list_plotted_density_evol(list_name, quadrant_sets)
    plt.savefig(save_path, dpi=200)
    print("Saved", save_path[-40:])
