"""Test using autofit etc to fit a very simple model to warm pixel-like data."""

import numpy as np
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

import hst_utilities as ut

path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(path, "../arctic/"))
import arcticpy as cti

sys.path.append(os.path.join(path, "../PyAutoArray/"))
import autoarray as aa

sys.path.append(os.path.join(path, "../PyAutoFit/"))
import autofit as af

sys.path.append(os.path.join(path, "../PyAutoCTI/"))
import autocti as ac


def single_pixel_fit_trap_density_and_time():
    """Fit a single trap's density and timescale for a single short trail"""
    # CTI model
    roe_in = cti.ROE()
    ccd_in = cti.CCD(full_well_depth=1e4, well_notch_depth=0.0, well_fill_power=1.0)
    traps_in = [
        cti.TrapInstantCapture(density=10.0, release_timescale=-1.0 / np.log(0.5))
    ]
    express = 0

    # Test input
    length = 9
    warm_index = 4
    pre_cti_in = np.zeros((length, 1))
    pre_cti_in[warm_index, 0] = 1000

    # Add CTI
    post_cti_in = cti.add_cti(
        image=pre_cti_in,
        parallel_roe=roe_in,
        parallel_ccd=ccd_in,
        parallel_traps=traps_in,
        parallel_express=express
    )

    # 1D lines for fitting
    pre_cti = pre_cti_in.T[0]
    post_cti = post_cti_in.T[0]
    print("Pre-CTI:", pre_cti)
    print("Post-CTI:", post_cti)

    # Test noise
    noise = np.ones_like(post_cti) * 0.01

    # Conversion value for HST ACS
    pixel_scale = 0.05

    # Convert to AutoCTI objects
    post_cti = ac.Array1D.manual_native(array=post_cti, pixel_scales=pixel_scale)
    noise = ac.Array1D.manual_native(array=noise, pixel_scales=pixel_scale)
    pre_cti = ac.Array1D.manual_native(array=pre_cti, pixel_scales=pixel_scale)

    shape_native = (length, 1)
    region_1d_list = [(warm_index, warm_index + 1)]
    normalization = np.sum(pre_cti)
    layout = ac.Layout1DLine(
        shape_1d=shape_native, region_list=region_1d_list, normalization=normalization
    )

    dataset_line = ac.DatasetLine(
        data=post_cti, noise_map=noise, pre_cti_data=pre_cti, layout=layout
    )

    # Set up fitting CTI model
    clocker = ac.Clocker1D(express=express)

    trap_0 = af.Model(ac.TrapInstantCapture)
    trap_0.density = af.UniformPrior(lower_limit=0.0, upper_limit=20.0)
    trap_0.release_timescale = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    trap_0.fractional_volume_none_exposed = 0.0
    trap_0.fractional_volume_full_exposed = 0.0
    trap_list = [trap_0]

    ccd = af.Model(ac.CCDPhase)
    ccd.well_notch_depth = ccd_in.phases[0].well_notch_depth
    ccd.well_fill_power = ccd_in.phases[0].well_fill_power
    ccd.full_well_depth = ccd_in.phases[0].full_well_depth

    model = af.Collection(cti=af.Model(ac.CTI1D, traps=trap_list, ccd=ccd))

    # Run the fit, currently writes the output to file
    search = af.DynestyStatic(path_prefix="dynesty/", name="test", nlive=10)
    analysis = ac.AnalysisDatasetLine(dataset_line=dataset_line, clocker=clocker)
    result = search.fit(model=model, analysis=analysis)

    # Extract results
    fitted_model = result.max_log_likelihood_instance.cti
    density = fitted_model.traps[0].density
    release_timescale = fitted_model.traps[0].release_timescale

    print("\nDensity:", density)
    print("Release timescale:", release_timescale)

    # Add CTI using fitted model to check
    post_cti_fit = cti.add_cti(
        image=pre_cti_in,
        parallel_roe=roe_in,
        parallel_ccd=ccd_in,
        parallel_traps=[
            cti.TrapInstantCapture(density=density, release_timescale=release_timescale)
        ],
        parallel_express=express
    )

    print("Data:", post_cti.T)
    print("Fit:", post_cti_fit.T[0])


def single_pixel_fit_3_traps_only_densities():
    """Fit three traps' densities with fixed timescales for a single trail at a
    set row from the readout register.
    """
    # CTI model
    roe_in = cti.ROE()
    ccd_in = cti.CCD(full_well_depth=84700, well_notch_depth=0.0, well_fill_power=0.478)
    relative_densities = np.array([0.17, 0.45, 0.38])
    trap_densities = relative_densities * 2
    release_times = np.array([0.74, 7.70, 37.0])
    traps_in = [
        cti.TrapInstantCapture(
            density=trap_densities[i], release_timescale=release_times[i]
        )
        for i in range(len(trap_densities))
    ]
    express = 0

    # Test input
    row = 500
    length = row + ut.trail_length
    warm_index = row
    pre_cti_in = np.zeros((length, 1))
    pre_cti_in[warm_index, 0] = 5000
    background = 50
    pre_cti_in += background

    # Add CTI
    post_cti_in = cti.add_cti(
        image=pre_cti_in,
        parallel_roe=roe_in,
        parallel_ccd=ccd_in,
        parallel_traps=traps_in,
        parallel_express=express
    )

    # 1D lines for fitting
    pre_cti = pre_cti_in.T[0]
    post_cti = post_cti_in.T[0]

    # Test noise
    noise = (np.random.rand(len(pre_cti)) + 1) * 5

    # Conversion value for HST ACS
    pixel_scale = 0.05

    # Convert to AutoCTI objects
    post_cti = ac.Array1D.manual_native(array=post_cti, pixel_scales=pixel_scale)
    noise = ac.Array1D.manual_native(array=noise, pixel_scales=pixel_scale)
    pre_cti = ac.Array1D.manual_native(array=pre_cti, pixel_scales=pixel_scale)

    shape_native = (length, 1)
    region_1d_list = [(warm_index, warm_index + 1)]
    normalization = np.sum(pre_cti)
    layout = ac.Layout1DLine(
        shape_1d=shape_native, region_list=region_1d_list, normalization=normalization
    )

    dataset_line = ac.DatasetLine(
        data=post_cti, noise_map=noise, pre_cti_data=pre_cti, layout=layout
    )

    # Set up fitting CTI model
    clocker = ac.Clocker1D(express=express)

    trap_list = []
    for trap in traps_in:
        trap_i = af.Model(ac.TrapInstantCapture)
        trap_i.density = af.UniformPrior(lower_limit=0.0, upper_limit=20.0)
        trap_i.release_timescale = trap.release_timescale
        trap_i.fractional_volume_none_exposed = 0.0
        trap_i.fractional_volume_full_exposed = 0.0
        trap_list.append(trap_i)
    # Fix relative densities, only fit the total            ###error
    # tot = trap_list[0].density / relative_densities[0]
    # trap_list[1].density = tot * relative_densities[1]
    # trap_list[2].density = tot * relative_densities[2]

    ccd = af.Model(ac.CCDPhase)
    ccd.well_notch_depth = ccd_in.phases[0].well_notch_depth
    ccd.well_fill_power = ccd_in.phases[0].well_fill_power
    ccd.full_well_depth = ccd_in.phases[0].full_well_depth

    model = af.Collection(cti=af.Model(ac.CTI1D, traps=trap_list, ccd=ccd))

    # Run the fit, currently writes the output to file
    search = af.DynestyStatic(path_prefix="dynesty/", name="test", nlive=20)
    analysis = ac.AnalysisDatasetLine(dataset_line=dataset_line, clocker=clocker)
    result = search.fit(model=model, analysis=analysis)

    # Extract results
    fitted_model = result.max_log_likelihood_instance.cti
    density_a = fitted_model.traps[0].density
    density_b = fitted_model.traps[1].density
    density_c = fitted_model.traps[2].density

    print("\nDensities in:", trap_densities[0], trap_densities[1], trap_densities[2])
    print("Densities fit:", density_a, density_b, density_c)

    # Add CTI using fitted model to check
    post_cti_fit = cti.add_cti(
        image=pre_cti_in,
        parallel_roe=roe_in,
        parallel_ccd=ccd_in,
        parallel_traps=[
            cti.TrapInstantCapture(
                density=density_c, release_timescale=release_times[0]
            ),
            cti.TrapInstantCapture(
                density=density_b, release_timescale=release_times[1]
            ),
            cti.TrapInstantCapture(
                density=density_c, release_timescale=release_times[2]
            ),
        ],
        parallel_express=express
    )

    print("\nTrail in:", post_cti.T[row - 1 :])
    print("Trail fit:", post_cti_fit.T[0][row - 1 :])


if __name__ == "__main__":

    # single_pixel_fit_trap_density_and_time()
    single_pixel_fit_3_traps_only_densities()
