"""Primary and plotting functions for hst_warm_pixels.py"""

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
