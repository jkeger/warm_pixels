"""Primary and plotting functions for hst_warm_pixels.py"""
import logging
import warnings

import autoarray as aa
import lmfit
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

import arcticpy
import arcticpy as cti
from warm_pixels import hst_utilities as ut
from warm_pixels import misc
from warm_pixels.misc import plot_hist
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection

logger = logging.getLogger(
    __name__
)


# ========
# Main functions
# ========
def find_consistent_warm_pixels(dataset, quadrant, flux_min=None, flux_max=None):
    """Find the consistent warm pixels in a dataset.

    find_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrant : str (opt.)
        The quadrant (A, B, C, D) of the image to load.

    flux_min, flux_max : float (opt.)
        If provided, then before checking for consistent pixels, discard any
        with fluxes outside of these limits.

    Saves
    -----
    warm_pixels : PixelLineCollection
        The set of consistent warm pixel trails, saved to
        dataset.saved_consistent_lines().
    """
    # Load
    warm_pixels = PixelLineCollection()
    warm_pixels.load(dataset.saved_lines(quadrant))

    # Ignore warm pixels below the minimum flux
    if flux_min is not None:
        sel_flux = np.where(
            (warm_pixels.fluxes > flux_min) & (warm_pixels.fluxes < flux_max)
        )[0]
        print("Kept %d bounded fluxes of %d" % (len(sel_flux), warm_pixels.n_lines))
        warm_pixels.lines = warm_pixels.lines[sel_flux]
        print("    ", end="")

    # Find the warm pixels present in at least e.g. 2/3 of the images
    consistent_lines = warm_pixels.find_consistent_lines(
        fraction_present=ut.fraction_present
    )
    print(
        "Found %d consistents of %d possibles"
        % (len(consistent_lines), warm_pixels.n_lines)
    )

    # Extract the consistent warm pixels
    warm_pixels.lines = warm_pixels.lines[consistent_lines]

    # Save
    warm_pixels.save(dataset.saved_consistent_lines(quadrant))


def extract_consistent_warm_pixels_corrected(dataset, quadrant):
    """Extract the corresponding warm pixels from the corrected images with CTI
    removed, in the same locations as the orignal consistent warm pixels.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrant : str (opt.)
        The quadrant (A, B, C, D) of the image to load.

    Saves
    -----
    warm_pixels_cor : PixelLineCollection
        The set of consistent warm pixel trails, saved to
        dataset.saved_consistent_lines(use_corrected=True).
    """
    # Load original warm pixels for the whole dataset
    warm_pixels = PixelLineCollection()
    warm_pixels.load(dataset.saved_consistent_lines(quadrant))

    # Corrected images
    warm_pixels_cor = PixelLineCollection()
    for i_image in range(dataset.n_images):
        image_path = dataset.cor_paths[i_image]
        image_name = dataset.image_names[i_image]
        print(
            "\r    %s_cor_%s (%d of %d) "
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

        # Select consistent warm pixels found from this image
        image_name_q = image_name + "_%s" % quadrant
        sel = np.where(warm_pixels.origins == image_name_q)[0]
        for i in sel:
            line = warm_pixels.lines[i]
            row, column = line.location

            # Copy the original metadata but take the data from the corrected image
            warm_pixels_cor.append(
                PixelLine(
                    data=image[
                         row - ut.trail_length: row + ut.trail_length + 1, column
                         ],
                    origin=line.origin,
                    location=line.location,
                    date=line.date,
                    background=line.background,
                )
            )

    print("Extracted %d lines" % warm_pixels_cor.n_lines)

    # Save
    warm_pixels_cor.save(dataset.saved_consistent_lines(quadrant, use_corrected=True))


def stack_dataset_warm_pixels(dataset, quadrants, use_corrected=False):
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

    use_corrected : bool (opt.)
        If True, then use the corrected images with CTI removed instead.

    Saves
    -----
    stacked_lines : PixelLineCollection
        The set of stacked pixel trails, saved to dataset.saved_stacked_lines().
    """
    # Load
    warm_pixels = PixelLineCollection()
    # Append data from each quadrant
    for quadrant in quadrants:
        warm_pixels.load(dataset.saved_consistent_lines(quadrant, use_corrected))

    # Subtract preceeding pixels in each line before stacking
    #
    # RJM - global background estimates may not be accurate, so keeping this information at this stage.
    #       It is always possible to do this subtraction after stacking.
    #
    # for i in range(warm_pixels.n_lines):
    #    warm_pixels.lines[i].data[ut.trail_length + 1 :] -= warm_pixels.lines[i].data[
    #        : ut.trail_length
    #    ][::-1]

    # Stack the lines in bins by distance from readout and total flux
    (
        stacked_lines,
        row_bins,
        flux_bins,
        date_bins,
        background_bins,
    ) = warm_pixels.generate_stacked_lines_from_bins(
        n_row_bins=ut.n_row_bins,
        flux_bins=ut.flux_bins,
        n_background_bins=ut.n_background_bins,
        return_bin_info=True,
    )
    print(
        "Stacked lines in %d bins"
        % (ut.n_row_bins * ut.n_flux_bins * ut.n_background_bins)
    )

    # Save
    stacked_lines.save(dataset.saved_stacked_lines(quadrants, use_corrected))
    np.savez(
        dataset.saved_stacked_info(quadrants, use_corrected),
        row_bins,
        flux_bins,
        date_bins,
        background_bins,
    )


def trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c):
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

    beta : float
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
    # print(n_bg,n_e)
    notch = 0
    return (
            rho_q
            * (((n_e - notch) / (w - notch)) ** beta - ((n_bg - notch) / (w - notch)) ** beta)
            * row
            * (
                    A * np.exp((1 - x) / tau_a) * (1 - np.exp(-1 / tau_a))
                    + B * np.exp((1 - x) / tau_b) * (1 - np.exp(-1 / tau_b))
                    + C * np.exp((1 - x) / tau_c) * (1 - np.exp(-1 / tau_c))
            )
        # + n_bg
    )


def trail_model_arctic(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c):
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

    beta : float
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
    # Set up classes required to run arCTIc
    # roe, ccd, traps = ac.CTI_model_for_HST_ACS(date)
    traps = [
        arcticpy.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
        arcticpy.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
        arcticpy.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
    ]
    roe = arcticpy.ROE()
    ccd = arcticpy.CCD(full_well_depth=w, well_fill_power=beta)

    # Work out how many trails are concatenated within the inputs
    trail_length = np.int(np.max(x))
    n_trails = x.size // trail_length

    # Loop over all those trails, to calculate the corresponding model
    output_model = np.zeros(n_trails * trail_length)
    for i in np.arange(n_trails):
        # Define input trail model, in format that can be passed to arCTIc
        warm_pixel_position = np.int(np.floor(row[i * trail_length]))
        warm_pixel_flux = n_e[i * trail_length]
        background_flux = n_bg[i * trail_length]
        model_before_trail = np.full(warm_pixel_position + 1 + trail_length, background_flux)
        model_before_trail[warm_pixel_position] = warm_pixel_flux

        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = arcticpy.add_cti(
            model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5
        ).flatten()  # convert back to a 1D array
        # print(model_after_trail[-15:])
        eper = model_after_trail[-trail_length:] - background_flux
        output_model[i * trail_length:(i + 1) * trail_length] = eper

    exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model


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
    beta = 0.478
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

    return trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)


def trail_model_hst_arctic(x, rho_q, n_e, n_bg, row, date):
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
    beta = 0.478
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

    # model_arctic = trail_model_arctic(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # model_exponentials = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(model_arctic[-24:])
    # print(model_exponentials[-24:])
    # print((model_arctic-model_exponentials)[-24:])
    # print()

    return trail_model_arctic(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)


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
            func=trail_model_hst_arctic, independent_vars=["x", "n_e", "n_bg", "row", "date"]
        )
    else:
        model = lmfit.models.Model(
            func=trail_model_hst, independent_vars=["x", "n_e", "n_bg", "row", "date"]
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
        dataset, quadrants, use_corrected=False, use_arctic=False,
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
    stacked_lines.load(dataset.saved_stacked_lines(quadrants, use_corrected))
    npzfile = np.load(dataset.saved_stacked_info(quadrants, use_corrected))
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


def cti_model_hst(date):
    """
    Return arcticpy objects that provide a preset CTI model for the Hubble Space
    Telescope (HST) Advanced Camera for Surveys (ACS).

    The returned objects are ready to be passed to add_cti() or remove_cti(),
    for parallel clocking.

    See Massey et al. (2014). Updated model and references coming soon.

    Parameters
    ----------
    date : float
        The Julian date. Should not be before the ACS launch date.

    Returns
    -------
    roe : ROE
        The ROE object that describes the read-out electronics.

    ccd : CCD
        The CCD object that describes how electrons fill the volume.

    traps : [Trap]
        A list of trap objects that set the parameters for each trap species.
    """
    assert date >= ut.date_acs_launch, "Date must be after ACS launch (2002/03/01)"

    # Trap species
    relative_densities = np.array([0.17, 0.45, 0.38])
    if date < ut.date_T_change:
        release_times = np.array([0.48, 4.86, 20.6])
    else:
        release_times = np.array([0.74, 7.70, 37.0])

    # Density evolution
    if date < ut.date_sm4_repair:
        initial_total_trap_density = -0.020
        trap_growth_rate = 4.22e-4
    else:
        initial_total_trap_density = -0.261
        trap_growth_rate = 5.55e-4
    total_trap_density = initial_total_trap_density + trap_growth_rate * (
            date - ut.date_acs_launch
    )
    trap_densities = relative_densities * total_trap_density

    # arctic objects
    roe = cti.ROE(
        dwell_times=[1.0],
        empty_traps_between_columns=True,
        empty_traps_for_first_transfers=False,
        force_release_away_from_readout=True,
        use_integer_express_matrix=False,
    )

    # Single-phase CCD
    ccd = cti.CCD(full_well_depth=84700, well_notch_depth=0.0, well_fill_power=0.478)

    # Instant-capture traps
    traps = [
        cti.TrapInstantCapture(
            density=trap_densities[i], release_timescale=release_times[i]
        )
        for i in range(len(trap_densities))
    ]

    return roe, ccd, traps


def remove_cti_dataset(dataset):
    """Remove CTI trails using arctic from all images in the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    Saves
    -----
    dataset.cor_paths
        The corrected images with CTI removed in the same location as the
        originals.
    """
    # Remove CTI from each image
    for i, image in enumerate(dataset.images):
        image_path = image.path
        image_name = image.name
        cor_path = image.cor_path
        print(
            "  Correcting %s (%d of %d)... "
            % (image_name, i + 1, len(dataset)),
            end="",
            flush=True,
        )

        # Load each quadrant of the image
        image_A, image_B, image_C, image_D = [
            aa.acs.ImageACS.from_fits(
                file_path=image_path,
                quadrant_letter=quadrant,
                bias_subtract_via_bias_file=True,
                bias_subtract_via_prescan=True,
            ).native
            for quadrant in ["A", "B", "C", "D"]
        ]

        # CTI model
        date = 2400000.5 + image_A.header.modified_julian_date
        roe, ccd, traps = cti_model_hst(date)

        def remove_cti(image):
            return cti.remove_cti(
                image=image,
                n_iterations=5,
                parallel_roe=roe,
                parallel_ccd=ccd,
                parallel_traps=traps,
                parallel_express=5
            )

        # Remove CTI (only print first time)
        if i == 0:
            print("")
            image_out_A = remove_cti(image_A)
            image_out_B, image_out_C, image_out_D = [
                remove_cti(image) for image in [image_B, image_C, image_D]
            ]
        else:
            image_out_A, image_out_B, image_out_C, image_out_D = [
                remove_cti(image) for image in [image_A, image_B, image_C, image_D]
            ]

        # Save the corrected image
        aa.acs.output_quadrants_to_fits(
            file_path=cor_path,
            quadrant_a=image_out_A,
            quadrant_b=image_out_B,
            quadrant_c=image_out_C,
            quadrant_d=image_out_D,
            header_a=image_A.header,
            header_b=image_B.header,
            header_c=image_C.header,
            header_d=image_D.header,
            overwrite=True,
        )

        print("Saved %s" % cor_path[-30:])


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
        colours = misc.A1_c[: len(quadrants)]
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

    misc.nice_plot(ax1)
    misc.nice_plot(ax2)
    misc.nice_plot(ax3)
    misc.nice_plot(ax4)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
        print("Saved", save_path[-36:])


def plot_stacked_trails(dataset, quadrants, use_corrected=False, save_path=None):
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
    # Load
    stacked_lines = PixelLineCollection()
    stacked_lines.load(dataset.saved_stacked_lines(quadrants, use_corrected))
    npzfile = np.load(dataset.saved_stacked_info(quadrants, use_corrected))
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
        dataset, quadrants, use_corrected=use_corrected, use_arctic=False
    )
    print(rho_q_set, rho_q_std_set, "exponentials")

    # Fit the total trap density to the full dataset using arCTIc
    print("Performing global fit")
    rho_q_set, rho_q_std_set, y_fit = fit_dataset_total_trap_density(
        dataset, quadrants, use_corrected=use_corrected, use_arctic=True
    )
    print(rho_q_set, rho_q_std_set, "ArCTIc")

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
                    n_row_bins=n_row_bins,
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
                    dataset, quadrants, use_corrected=use_corrected, use_arctic=True,
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
                    date=dataset.date,
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
                        r"Hot pixel (e$^-$):",
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
        print("Saved", save_path[-40:])


def plot_trap_density_evol(
        list_name, quadrant_sets, do_sunspots=True, use_corrected=False, do_pdf=False
):
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

    use_corrected : bool (opt.)
        If True, then also plot the results from the corrected images with CTI
        removed.

    do_pdf : bool (opt.)
        If True, then save as a pdf instead of a png.
    """
    # Colours
    if len(quadrant_sets) == 1:
        colours = ["k"]
        colours_cor = ["0.35"]
    else:
        colours = misc.A1_c[: len(quadrant_sets)]
        colours_cor = misc.A1_c[: len(quadrant_sets)]

    # Set date limits
    npzfile = np.load(ut.dataset_list_saved_density_evol(list_name, quadrant_sets[0]))
    days, densities, errors = [npzfile[var] for var in npzfile.files]
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
    for i_q, quadrants in enumerate(quadrant_sets):
        # Load
        npzfile = np.load(ut.dataset_list_saved_density_evol(list_name, quadrants))
        days, densities, errors = [npzfile[var] for var in npzfile.files]

        label = "".join(quadrants)
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

            # Load
            npzfile = np.load(
                ut.dataset_list_saved_density_evol(list_name, quadrants, use_corrected)
            )
            days, densities_cor, errors_cor = [npzfile[var] for var in npzfile.files]

            # Plot negative values separately
            where_pos = np.where(densities_cor > 0)[0]
            where_neg = np.where(densities_cor < 0)[0]

            # Data
            ax.errorbar(
                days[where_pos],
                densities_cor[where_pos],
                yerr=errors_cor[where_pos],
                c=c,
                ls="none",
                marker="x",
                capsize=3,
                elinewidth=1,
                label="After correction",
            )
            ax.scatter(
                days[where_neg],
                abs(densities_cor[where_neg]),
                color=c,
                facecolor="w",
                marker="o",
                zorder=-2,
            )
            ax.errorbar(
                days[where_neg],
                abs(densities_cor[where_neg]),
                yerr=errors_cor[where_neg],
                c=c,
                ls="none",
                capsize=3,
                elinewidth=1,
                zorder=-1,
            )

    # ========
    # HST CTI measurements using Richard's IDL code
    # ========
    if not True:  ##
        # date, density, density_err
        data = np.array(
            [
                [431.303, 0.179387, 0.0682717],  # shortSNe2
                [804.024, 0.325217, 0.0512948],  # 05_2004
                [1131.27, 0.456763, 0.762311],  # 04_2005
                [1519.10, 0.627182, 0.0732714],  # 04_2006
                [1599.39, 0.611703, 0.0760443],  # richmassey60490
                [1613.18, 0.560601, 0.0496126],  # richmassey61093
                [1629.13, 0.632204, 0.0515503],  # richmassey60491
                [1655.14, 0.657068, 0.0503882],  # richmassey61092
                [2803.10, 1.34501, 0.0720851],  # sm43
                [3007.13, 1.45635, 0.0732634],  # 05_2010
                [3321.37, 1.65278, 0.0453292],  # 04_2011
                [3799.49, 1.89259, 0.0684670],  # huff_spt814b
                [4050.26, 2.01314, 0.0802822],  # 04_2013
                [4377.37, 2.07898, 0.0479423],  # 02_2014
                [4709.00, 2.29900, 0.238915],  # 01_2015
                [5058.00, 2.48080, 0.297159],  # 01_2016
                [5514.32, 2.69825, 0.0761266],  # 04_2017
                [5695.42, 2.58939, 0.0724275],  # 10_2017
                [6008.27, 2.84505, 0.351008],  # 08_2018
                [6240.09, 3.01478, 0.0649324],  # 04_2019
                [6595.34, 3.16847, 0.606145],  # 03_2020
                [6852.48, 3.26501, 0.209639],  # 12_2020
            ]
        )
        ax.errorbar(
            data[:, 0],
            data[:, 1],
            yerr=data[:, 2],
            ls="none",
            marker="+",
            capsize=3,
            elinewidth=1,
        )

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
        list_name, quadrant_sets, do_pdf=do_pdf
    )
    plt.savefig(save_path, dpi=200)
    print("Saved", save_path[-40:])
