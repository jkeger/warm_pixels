import numpy as np
import time

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
    roe = cti.ROE()
    ccd = cti.CCD(full_well_depth=w, well_fill_power=beta)

    # Work out how many trails are concatenated within the inputs
    trail_length = np.int(np.max(x)) #Find max number of 0,1,2,3...12
    print ('Trail length is', trail_length)
    n_trails = x.size // trail_length #Divide all the x's by 12
    print ('Total number of trails', n_trails)

    output_model = np.zeros(n_trails * trail_length) #empty array 
    
    # Loop over all those trails, to calculate the corresponding model
    print('Extracting the data to feed into arctic...')
    extract_start=time.time()
    warm_pixel_position_all=[]
    warm_pixel_flux_all=[]
    background_flux_all=[]
    model_before_trail_all=[]
  
    for i in np.arange(n_trails):
        warm_pixel_position = np.int(np.floor(row[i * trail_length]))
        warm_pixel_position_all.append(warm_pixel_position)
        warm_pixel_flux = n_e[i * trail_length]
        warm_pixel_flux_all.append(warm_pixel_flux)
        background_flux = n_bg[i * trail_length]
        background_flux_all.append(background_flux)
        model_before_trail = np.full(warm_pixel_position + 1 + trail_length, background_flux)
        model_before_trail[warm_pixel_position] = warm_pixel_flux
        model_before_trail_all.append(model_before_trail)
    extract_end=time.time()
    print('Total extraction time:', extract_end-extract_start)
        
    model_after_trail_all=[]
    fit_start=time.time()
    #for i in np.arange(n_trails):
    for lines in model_before_trail_all:
        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = cti.add_cti(
            lines.reshape(-1, 1),  # pass 2D image to arCTIc
            parallel_roe=roe,
            parallel_ccd=ccd,
            parallel_traps=traps,
            parallel_express=5
        ).flatten()  # convert back to a 1D array
        model_after_trail_all.append(model_after_trail)
        # print(model_after_trail[-15:])

    fit_end=time.time()
    print('Total fit time:', fit_end-fit_start)

    
    start_eper=time.time()
    for i in np.arange(n_trails):
        eper = model_after_trail_all[i][-trail_length:] - background_flux_all[i]
        output_model[i * trail_length:(i + 1) * trail_length] = eper 
    end_eper=time.time()
    print('Total EPER construction time:', end_eper-start_eper)

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
