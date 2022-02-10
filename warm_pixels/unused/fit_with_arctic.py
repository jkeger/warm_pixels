import autofit as af
import numpy as np

import arcticpy
from warm_pixels.hst_data import *  # Would be nice not to have to do this
from warm_pixels.pixel_lines import PixelLineCollection

"""
Fit warm pixel data sets using arCTIc. Can be called from standard hst_warm_pixels.py,
but also works as a minimal standalone I/O and fitter. For that, run from a python console:
>>> import fit_with_arctic as ff
>>> ff.fit_warm_pixels_with_arctic()
This requires the following files to exist, which were created by hst_warm_pixels.py
../hst_acs_datasets/07_2020/saved_stacked_info_ABCD.npz
../hst_acs_datasets/07_2020/saved_stacked_lines_ABCD.pickle
"""


def fit_warm_pixels_with_arctic(
        dataset=None, quadrants="ABCD", use_corrected=False,
        row_bins=None, flux_bins=None, background_bins=None
):
    # Read in trail data
    if dataset is None:
        dataset = Dataset("07_2020")
    eper_trail, eper_noise, eper_untrailed = load_trails(dataset, quadrants,
                                                         use_corrected=use_corrected,
                                                         row_bins=row_bins, flux_bins=flux_bins,
                                                         background_bins=background_bins
                                                         )
    # Define priors
    model = af.Model(CTImodel)
    model.trap_density_A = af.UniformPrior(lower_limit=0.0, upper_limit=5.0)
    # model.trap_density_A = af.GaussianPrior(mean=0.6, sigma=0.1)
    model.trap_release_time_A = af.GaussianPrior(mean=0.74, sigma=0.1)  # Want this to be >0. Possibly lognormal
    model.add_assertion(model.trap_release_time_A > 0.0)
    # model.trap_density_B = af.UniformPrior(lower_limit=0.5, upper_limit=5.0)
    # model.trap_release_time_B = af.GaussianPrior(mean=7.7, sigma=0.5) # Want this to be >0. Possibly lognormal
    # model.add_assertion(model.trap_release_time_B > model.trap_release_time_A)
    # model.warm_pixel_flux = af.LogUniformPrior(lower_limit=1.0, upper_limit=1.0e7) # Want this to be lognormal
    # model.add_assertion(model.trap_density_A > 0.0)

    # How do I initialise search at a good guess (particularly for warm_pixel_flux)?

    # Fit trail data (no longer need to pass model_untrailed into this)
    analysis = Analysis(data_trailed=eper_trail, noise=eper_noise, data_untrailed=eper_untrailed)
    # emcee = af.Emcee(nwalkers=50, nsteps=2000)
    # result = emcee.fit(model=model, analysis=analysis)
    dynesty = af.DynestyStatic(
        # name="cti_test_dynesty2",
        nlive=50,
        sample="rwalk",
        iterations_per_update=2500,
        # number_of_cores=6
    )
    result = dynesty.fit(model=model, analysis=analysis)
    print(result)

    return result


def load_trails(
        dataset, quadrants, use_corrected=False,
        row_bins=None, flux_bins=None, background_bins=None
):
    """Read in warm pixel trails and reformat them to pass to arCTIc

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
    model_trail : [ numpy 1D float arrays ]
        A list of the number of electrons in a CCD column, after CTI trailing (EPER).

    model_noise : [ numpy 1D float arrays ]
        A list of the uncertainty on the number of electrons in a CCD column, after CTI trailing.
    """

    # Load data
    stacked_lines = PixelLineCollection()
    stacked_lines.load(dataset.saved_stacked_lines(quadrants, use_corrected))
    npzfile = np.load(dataset.saved_stacked_info(quadrants, use_corrected))
    n_row_bins, n_flux_bins, n_date_bins, n_background_bins = [
        (len(npzfile[var]) - 1) for var in npzfile.files
    ]

    # Parse inputs, to decide which bin to fit
    if row_bins is None:
        row_bins = range(n_row_bins)
    if flux_bins is None:
        flux_bins = range(n_flux_bins)
    if background_bins is None:
        background_bins = range(n_background_bins)

    # Compile each trail to be fitted into a list of trails
    n_lines_used = 0  # Counter
    model_trail = []  # Initialise lists, to be added to in turn
    model_noise = []
    model_untrailed = []
    for i_row in row_bins:
        for i_flux in flux_bins:
            for i_background in background_bins:

                # If only using some bins, decide which ones to
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
                    model_trail.append(line.model_full_trail)
                    model_noise.append(line.model_full_trail_noise)
                    model_untrailed.append(line.model_full_trail_untrailed)
                    n_lines_used += 1

    if n_lines_used == 0:
        return None, None, None

    return model_trail, model_noise, model_untrailed


class CTImodel:
    def __init__(
            self,
            trap_density_A=1.0,
            trap_release_time_A=1.0,
            # trap_density_B=1.0,
            # trap_release_time_B=1.0,
            # warm_pixel_flux=1.0,
    ):
        self.trap_density_A = trap_density_A
        self.trap_release_time_A = trap_release_time_A
        # self.trap_density_B = trap_density_B
        # self.trap_release_time_B = trap_release_time_B
        # self.warm_pixel_flux = warm_pixel_flux

    """
    A function to turn input values into an EPER trail.
    An instance of this class will be available during model fitting.
    This method will be used to fit the model to data and compute a likelihood.
    """

    def add_trail_to_ccd_columns(self, data_untrailed):

        # Set up classes required to run arCTIc
        # date = 2400000.5 + image_A.header.modified_julian_date
        # roe, ccd, traps = ac.CTI_model_for_HST_ACS(date)
        # Or manual CTI model  (see class docstrings in src/<traps,roe,ccd>.cpp)
        traps = [
            # arcticpy.TrapInstantCapture(density=self.trap_density_A, release_timescale=self.trap_release_time_A),
            # arcticpy.TrapInstantCapture(density=self.trap_density_B, release_timescale=self.trap_release_time_B),
            # arcticpy.TrapInstantCapture(density=self.trap_density_C, release_timescale=self.trap_release_time_C),
            arcticpy.TrapInstantCapture(density=self.trap_density_A * 0.6 / 3.6,
                                        release_timescale=self.trap_release_time_A),
            arcticpy.TrapInstantCapture(density=self.trap_density_A * 1.6 / 3.6,
                                        release_timescale=self.trap_release_time_A * 7.7 / 0.74),
            arcticpy.TrapInstantCapture(density=self.trap_density_A * 1.4 / 3.6,
                                        release_timescale=self.trap_release_time_A * 37.0 / 0.74),
        ]
        roe = arcticpy.ROE()
        ccd = arcticpy.CCD(full_well_depth=84700, well_fill_power=0.478)

        # Run arCTIc to produce the output image with EPER trails
        model_after_trail = []
        for model_before_trail in data_untrailed:
            model_after_trail.append(
                arcticpy.add_cti(
                    model_before_trail.reshape(-1, 1),  # pass 2D image to arCTIc
                    parallel_roe=roe,
                    parallel_ccd=ccd,
                    parallel_traps=traps,
                    parallel_express=5
                ).flatten()  # convert back to a 1D array
            )

        return model_after_trail  # convert arCTIc output to a 1D row


class Analysis(af.Analysis):

    def __init__(self, data_trailed, noise, data_untrailed):
        self.data_trailed = data_trailed
        self.noise = noise
        self.data_untrailed = data_untrailed

    def log_likelihood_function(self, instance):
        """
        The 'instance' that comes into this method is an instance of the Gaussian class
        above, with the parameters set to values chosen by the non-linear search.
        """
        # if instance.trap_density_A < 0: return -np.inf
        # if instance.trap_release_time_A < 0: return -np.inf
        # if instance.warm_pixel_flux < 0: return -np.inf

        model_data_trailed = instance.add_trail_to_ccd_columns(data_untrailed=self.data_untrailed)

        # chi_squareds = []
        # for i in np.arange(len(self.data_trailed)):
        #    chi_squared = ( ( model_data_trailed[i] - self.data_trailed[i] ) / self.noise[i] ) ** 2
        #    chi_squareds.append( np.sum( chi_squared ) )
        # log_likelihood = -0.5 * sum(chi_squareds)

        chi_squareds = 0.
        for i in np.arange(len(self.data_trailed)):
            chi_squared = ((model_data_trailed[i] - self.data_trailed[i]) / self.noise[i]) ** 2
            chi_squareds += np.sum(chi_squared)
        log_likelihood = -0.5 * chi_squareds

        print("trap A: ", instance.trap_density_A, instance.trap_release_time_A, chi_squareds)

        return log_likelihood
