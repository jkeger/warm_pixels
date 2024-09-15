import numpy as np

import ArCTIcpy as cti
from warm_pixels import hst_utilities as ut


def trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c):
    """Calculate the model shape of a CTI trail as a sum of exponentials.

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
    notch = 0
    return (
        rho_q
        * (
            ((n_e - notch) / (w - notch)) ** beta
            - ((n_bg - notch) / (w - notch)) ** beta
        )
        * row
        * (
            A * np.exp((1 - x) / tau_a) * (1 - np.exp(-1 / tau_a))
            + B * np.exp((1 - x) / tau_b) * (1 - np.exp(-1 / tau_b))
            + C * np.exp((1 - x) / tau_c) * (1 - np.exp(-1 / tau_c))
        )
        # + n_bg ##
    )


def trail_model_ArCTIc(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c):
    """Calculate the model shape of a CTI trail using ArCTIc.

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
    # Set up classes required to run ArCTIc
    # roe, ccd, traps = ac.CTI_model_for_HST_ACS(date)
    traps = [
        cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
        cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
        cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
    ]
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta)

    # Work out how many trails are concatenated within the inputs
    trail_length = np.int(np.max(x))
    n_trails = x.size // trail_length

    # Loop over all those trails, to calculate the corresponding model
    output_model = np.zeros(n_trails * trail_length)
    for i in np.arange(n_trails):
        # Define input trail model, in format that can be passed to arctic
        warm_pixel_position = np.int(np.floor(row[i * trail_length]))
        warm_pixel_flux = n_e[i * trail_length]
        background_flux = n_bg[i * trail_length]
        model_before_trail = np.full(
            warm_pixel_position + 1 + trail_length, background_flux
        )
        model_before_trail[warm_pixel_position] = warm_pixel_flux

        # Run arctic to produce the output image with EPER trails
        model_after_trail = cti.add_cti(
            model_before_trail.reshape(-1, 1),  # 2D image for arctic
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5,
        ).flatten()  # convert back to a 1D array
        eper = model_after_trail[-trail_length:] - background_flux
        output_model[i * trail_length : (i + 1) * trail_length] = eper

    exponential_model = trail_model(
        x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c
    )

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
        tau_a = 0.48  ##store these somewhere?
        tau_b = 4.86
        tau_c = 20.6
    else:
        tau_a = 0.74
        tau_b = 7.70
        tau_c = 37.0

    return trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)


def trail_model_hst_ArCTIc(x, rho_q, n_e, n_bg, row, date):
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

    return trail_model_ArCTIc(
        x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c
    )
