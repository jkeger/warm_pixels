import numpy as np

import arcticpy as cti
from warm_pixels import hst_utilities as ut


#def trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c):
def trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c, N):
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
            * (((12*N + n_e - notch) / (w - notch)) ** beta - ((n_bg - notch - N) / (w - notch)) ** beta)
            * row
            * (
                    A * np.exp((1 - x) / tau_a) * (1 - np.exp(-1 / tau_a))
                    + B * np.exp((1 - x) / tau_b) * (1 - np.exp(-1 / tau_b))
                    + C * np.exp((1 - x) / tau_c) * (1 - np.exp(-1 / tau_c))
            ) 
            #+N
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
        cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
        cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
        cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
    ]
    
# =============================================================================
#     traps = [
#         cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
#         cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
#         #cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=1.0),
#         #cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=1.0),
#         cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=1.0),
#     ]
#     
# =============================================================================
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta)

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
        model_before_trail = np.full(warm_pixel_position + 1 + trail_length, background_flux) # move out of loop
        model_before_trail[warm_pixel_position] = warm_pixel_flux

        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = cti.add_cti(
            model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5,
            verbosity=0
        ).flatten()  # convert back to a 1D array
        # print(model_after_trail[-15:])
        eper = model_after_trail[-trail_length:] - background_flux 
        output_model[i * trail_length:(i + 1) * trail_length] = eper # put out of loop

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_notch(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c, notch):
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
        cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
        cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
        cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
    ]
    
# =============================================================================
#     traps = [
#         cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
#         cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
#         #cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=1.0),
#         #cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=1.0),
#         cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=1.0),
#     ]
#     
# =============================================================================
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta, well_notch_depth=notch)

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
        model_before_trail = np.full(warm_pixel_position + 1 + trail_length, background_flux) # move out of loop
        model_before_trail[warm_pixel_position] = warm_pixel_flux

        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = cti.add_cti(
            model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5,
            verbosity=0
        ).flatten()  # convert back to a 1D array
        # print(model_after_trail[-15:])
        eper = model_after_trail[-trail_length:] - background_flux 
        output_model[i * trail_length:(i + 1) * trail_length] = eper # put out of loop

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_notch_pushed(x, rho_q, generated_trails, beta, w, A, B, C, tau_a, tau_b, tau_c, notch):
    
    # Set up classes required to run arCTIc
    # roe, ccd, traps = ac.CTI_model_for_HST_ACS(date)
    traps = [
        cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
        cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
        cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
    ]
    
# =============================================================================
#     traps = [
#         cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
#         cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
#         #cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=1.0),
#         #cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=1.0),
#         cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=1.0),
#     ]
#     
# =============================================================================
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta, well_notch_depth=notch)

    # Work out how many trails are concatenated within the inputs
    trail_length = np.int(np.max(x))
    n_trails = x.size // trail_length

    # Loop over all those trails, to calculate the corresponding model
    output_model = np.zeros(n_trails * trail_length)
    for i in np.arange(n_trails):
        # Define input trail model, in format that can be passed to arCTIc
        model_before_trail = generated_trails[i]
        background_flux=model_before_trail[0]

        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = cti.add_cti(
            model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5,
            verbosity=0
        ).flatten()  # convert back to a 1D array
        # print(model_after_trail[-15:])
        eper = model_after_trail[-trail_length:] - background_flux 
        output_model[i * trail_length:(i + 1) * trail_length] = eper # put out of loop

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_notch_pushed_plot(x, rho_q, generated_trails, beta, w, A, B, C, tau_a, tau_b, tau_c, notch):
    
    # Set up classes required to run arCTIc
    # roe, ccd, traps = ac.CTI_model_for_HST_ACS(date)
    traps = [
        cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
        cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
        cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
    ]
    
# =============================================================================
#     traps = [
#         cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
#         cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
#         #cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=1.0),
#         #cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=1.0),
#         cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=1.0),
#     ]
#     
# =============================================================================
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta, well_notch_depth=notch)

    # Work out how many trails are concatenated within the inputs
    trail_length = np.int(np.max(x))
    n_trails = x.size // trail_length

    # Loop over all those trails, to calculate the corresponding model
    output_model = np.zeros(n_trails * trail_length)

    # Define input trail model, in format that can be passed to arCTIc
    model_before_trail = generated_trails
    background_flux=model_before_trail[0]

    # Run arCTIc to produce the output image with EPER trails
    model_after_trail = cti.add_cti(
        model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
        parallel_roe=roe,
        parallel_ccd=ccd,
        parallel_traps=traps,
        parallel_express=5,
        verbosity=0
    ).flatten()  # convert back to a 1D array
    # print(model_after_trail[-15:])
    eper = model_after_trail[-trail_length:] - background_flux 
    output_model[0 * trail_length:(0 + 1) * trail_length] = eper # put out of loop

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_notch_pushed_continuum(x, rho_q, generated_trails, beta, w, A, B, C, tau_a, tau_b, tau_c, sigma_a, sigma_b, sigma_c, notch):
    
    # Set up classes required to run arCTIc
    # roe, ccd, traps = ac.CTI_model_for_HST_ACS(date)
    traps = [
        cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=sigma_a),
        cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=sigma_b),
        cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=sigma_c),
    ]
    
# =============================================================================
#     traps = [
#         cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
#         cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
#         #cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=1.0),
#         #cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=1.0),
#         cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=1.0),
#     ]
#     
# =============================================================================
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta, well_notch_depth=notch)

    # Work out how many trails are concatenated within the inputs
    trail_length = np.int(np.max(x))
    n_trails = x.size // trail_length

    # Loop over all those trails, to calculate the corresponding model
    output_model = np.zeros(n_trails * trail_length)
    for i in np.arange(n_trails):
        # Define input trail model, in format that can be passed to arCTIc
        model_before_trail = generated_trails[i]
        background_flux=model_before_trail[0]

        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = cti.add_cti(
            model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5,
            verbosity=0
        ).flatten()  # convert back to a 1D array
        # print(model_after_trail[-15:])
        eper = model_after_trail[-trail_length:] - background_flux 
        output_model[i * trail_length:(i + 1) * trail_length] = eper # put out of loop

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_notch_pushed_plot_continuum(x, rho_q, generated_trails, beta, w, A, B, C, tau_a, tau_b, tau_c, sigma_a, sigma_b, sigma_c, notch):
    
    # Set up classes required to run arCTIc
    # roe, ccd, traps = ac.CTI_model_for_HST_ACS(date)
    traps = [
        cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=sigma_a),
        cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=sigma_b),
        cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=sigma_c),
    ]
    
# =============================================================================
#     traps = [
#         cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
#         cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
#         #cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=1.0),
#         #cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=1.0),
#         cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=1.0),
#     ]
#     
# =============================================================================
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta, well_notch_depth=notch)

    # Work out how many trails are concatenated within the inputs
    trail_length = np.int(np.max(x))
    n_trails = x.size // trail_length

    # Loop over all those trails, to calculate the corresponding model
    output_model = np.zeros(n_trails * trail_length)

    # Define input trail model, in format that can be passed to arCTIc
    model_before_trail = generated_trails
    background_flux=model_before_trail[0]

    # Run arCTIc to produce the output image with EPER trails
    model_after_trail = cti.add_cti(
        model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
        parallel_roe=roe,
        parallel_ccd=ccd,
        parallel_traps=traps,
        parallel_express=5,
        verbosity=0
    ).flatten()  # convert back to a 1D array
    # print(model_after_trail[-15:])
    eper = model_after_trail[-trail_length:] - background_flux 
    output_model[0 * trail_length:(0 + 1) * trail_length] = eper # put out of loop

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_continuum(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c, sigma_a, sigma_b, sigma_c):
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
# =============================================================================
#     traps = [
#         cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
#         cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
#         cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
#     ]
# =============================================================================
    
    traps = [
        cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=sigma_a),
        cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=sigma_b),
        cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=sigma_c),
    ]
    
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta)

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
        model_before_trail = np.full(warm_pixel_position + 1 + trail_length, background_flux) # move out of loop
        model_before_trail[warm_pixel_position] = warm_pixel_flux

        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = cti.add_cti(
            model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5,
            verbosity=0
        ).flatten()  # convert back to a 1D array
        # print(model_after_trail[-15:])
        eper = model_after_trail[-trail_length:] - background_flux 
        output_model[i * trail_length:(i + 1) * trail_length] = eper # put out of loop

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_continuum_notch(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c, sigma_a, sigma_b, sigma_c, notch):
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
# =============================================================================
#     traps = [
#         cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
#         cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
#         cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
#     ]
# =============================================================================
    
    traps = [
        cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=sigma_a),
        cti.TrapInstantCaptureContinuum(density=B * rho_q, release_timescale=tau_b, release_timescale_sigma=sigma_b),
        cti.TrapInstantCaptureContinuum(density=C * rho_q, release_timescale=tau_c, release_timescale_sigma=sigma_c),
    ]
    
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta, well_notch_depth=notch)

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
        model_before_trail = np.full(warm_pixel_position + 1 + trail_length, background_flux) # move out of loop
        model_before_trail[warm_pixel_position] = warm_pixel_flux

        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = cti.add_cti(
            model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5,
            verbosity=0
        ).flatten()  # convert back to a 1D array
        # print(model_after_trail[-15:])
        eper = model_after_trail[-trail_length:] - background_flux 
        output_model[i * trail_length:(i + 1) * trail_length] = eper # put out of loop

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_a_continuum(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c, sigma_a):
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
# =============================================================================
#     traps = [
#         cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
#         cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
#         cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
#     ]
# =============================================================================
    
    traps = [
        cti.TrapInstantCaptureContinuum(density=A * rho_q, release_timescale=tau_a, release_timescale_sigma=sigma_a),
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
        # Define input trail model, in format that can be passed to arCTIc
        warm_pixel_position = np.int(np.floor(row[i * trail_length]))
        warm_pixel_flux = n_e[i * trail_length]
        background_flux = n_bg[i * trail_length]
        model_before_trail = np.full(warm_pixel_position + 1 + trail_length, background_flux) # move out of loop
        model_before_trail[warm_pixel_position] = warm_pixel_flux

        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = cti.add_cti(
            model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5,
            verbosity=0
        ).flatten()  # convert back to a 1D array
        # print(model_after_trail[-15:])
        eper = model_after_trail[-trail_length:] - background_flux 
        output_model[i * trail_length:(i + 1) * trail_length] = eper # put out of loop

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_speed(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c):
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
        cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
        cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
        cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
    ]
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta)

    # Work out how many trails are concatenated within the inputs
    trail_length = np.int(np.max(x))
    n_trails = x.size // trail_length
    
    # For each trail, find the background and the warm pixel flux and position
    warm_pixel_positions=[]
    warm_pixel_fluxes=[]
    warm_pixel_backgrounds=[]
    for i in np.arange(n_trails):
        warm_pixel_positions.append(np.int(np.floor(row[i * trail_length])))
        warm_pixel_fluxes.append(n_e[i * trail_length])
        warm_pixel_backgrounds.append(n_bg[i * trail_length])
        
    # The farthest warm pixel determines the length of the columns for the input model    
    farthest_wp=max(warm_pixel_positions) 
    longest_column = farthest_wp + 1 + trail_length
    
    # Generate input model template and output model template
    input_model=np.zeros(shape=(longest_column, n_trails)) # 50 columns, each with length of about 2000 
    output_model = np.zeros(n_trails * trail_length) # Just a 1D array, same as what we feed this function
    
    # Fill in the input model template
    for i in np.arange(n_trails):
        input_model[:,i]=warm_pixel_backgrounds[i] # Each column should have the correct background
        input_model[warm_pixel_positions[i],i]=warm_pixel_fluxes[i] # Put the warm pixel in the correct spot
        
# =============================================================================
#     print(input_model)
#     #print(len(input_model[0]))
#     #print(len(input_model))
#     print(input_model[warm_pixel_positions[0]:warm_pixel_positions[0]+13,0])
# =============================================================================
    

        # Run arCTIc to produce the output image with EPER trails
    model_after_trail = cti.add_cti(
        #arctic_input.reshape(-1, 1),  # pass 2D image to arCTIc
        input_model,
        parallel_roe=roe,
        parallel_ccd=ccd,
        parallel_traps=traps,
        parallel_express=5
    )  
    
# =============================================================================
#     print(model_after_trail)
#     print(model_after_trail[warm_pixel_positions[0]:warm_pixel_positions[0]+13,0])
# =============================================================================

    # Fill in the output model template
    for i in np.arange(n_trails):
        eper = model_after_trail[warm_pixel_positions[i] + 1:warm_pixel_positions[i] + trail_length + 1,i] - warm_pixel_backgrounds[i]
        output_model[i * trail_length:(i + 1) * trail_length] = eper
        
    #print(output_model[0:15])

    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
    # print(output_model[-24:])
    # print(exponential_model[-24:])
    # print((output_model-exponential_model)[-24:])
    # print()

    return output_model

def trail_model_arctic_one(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c): # Wont work since trails
#overlap
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
        cti.TrapInstantCapture(density=A * rho_q, release_timescale=tau_a),
        cti.TrapInstantCapture(density=B * rho_q, release_timescale=tau_b),
        cti.TrapInstantCapture(density=C * rho_q, release_timescale=tau_c),
    ]
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta)

    # Work out how many trails are concatenated within the inputs
    trail_length = np.int(np.max(x))
    n_trails = x.size // trail_length

    # Generate a template for the output model 
    output_model = np.zeros(n_trails * trail_length)
    
    # Find furthest warm pixel position
    warm_pixel_positions=[]
    warm_pixel_fluxes=[]
    for i in np.arange(n_trails):
        warm_pixel_positions.append(np.int(np.floor(row[i * trail_length])))
        warm_pixel_fluxes.append(n_e[i * trail_length])
    farthest_wp=max(warm_pixel_positions)
    number_of_positions=len(warm_pixel_positions)
    
    # Generate big 1D array to be trailed 
    background_flux=n_bg[0] # they're all roughly the same so just pick one
    model_before_trail=np.full(farthest_wp + 1 + trail_length, background_flux)
    
    # Put the untrailed warm pixels into the big array
    for j in np.arange(number_of_positions):
        current_position=warm_pixel_positions[j]
        model_before_trail[current_position]=warm_pixel_fluxes[j]
    
    
    # Run arCTIc once to produce the output image with EPER trails for all 50 at the same time
    model_after_trail = cti.add_cti(
        model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
        parallel_roe=roe,
        parallel_ccd=ccd,
        parallel_traps=traps,
        parallel_express=5
    ).flatten()  # convert back to a 1D array
    # print(model_after_trail[-15:])
    
    # 
    for k in np.arange(n_trails):
        current_position=warm_pixel_positions[k]
        eper = model_after_trail[current_position+1:current_position+13] - background_flux
        output_model[k * trail_length:(k + 1) * trail_length] = eper 


    #exponential_model = trail_model(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c)
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
