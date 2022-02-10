"""Primary and plotting functions for hst_warm_pixels.py"""

import autoarray as aa
import numpy as np

import arcticpy as cti
from warm_pixels import hst_utilities as ut


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
        image_name = image.name
        print(
            "  Correcting %s (%d of %d)... "
            % (image_name, i + 1, len(dataset)),
            end="",
            flush=True,
        )

        # Load each quadrant of the image
        image_A, image_B, image_C, image_D = [
            image.load_quadrant(quadrant)
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
            file_path=image.corrected().path,
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

        print(f"Saved {image.corrected().path.stem}")
