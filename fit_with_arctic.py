from pixel_lines import PixelLine, PixelLineCollection
from hst_data import *
import arcticpy
import autofit as af
import numpy as np

def fit_warm_pixels_with_arctic(
        dataset=None, quadrants="ABCD", use_corrected=False,
        row_bins=None, flux_bins=None, background_bins=None
):
    # Read in trail data
    if dataset is None: dataset = Dataset("07_2020")
    model_trail, model_noise, model_untrailed = load_trails(dataset, quadrants,
                                                            use_corrected=use_corrected,
                                                            row_bins=row_bins, flux_bins=flux_bins,
                                                            background_bins=background_bins
                                                            )

    # Define priors
    print("Hello world 1")
    print(len(model_trail))
    print(model_trail[0].shape)
    model = af.Model(CTImodel)
    model.trap_density_A = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    model.trap_release_time_A = af.GaussianPrior(mean=10.0, sigma=5.0) # Want this to be >0. Possibly lognormal
    model.warm_pixel_flux = af.LogUniformPrior(lower_limit=1.0, upper_limit=1.0e7) # Want this to be lognormal

    # How to initialise search at a good guess (particularly for warm_pixel_flux)?

    # Fit trail data
    print("Hello world 2")
    analysis = Analysis(data=model_trail, noise_map=model_noise)
    emcee = af.Emcee(nwalkers=50, nsteps=2000)
    result = emcee.fit(model=model, analysis=analysis)

    print("Hello world 3")
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
    if row_bins is None: row_bins = range(n_row_bins)
    if flux_bins is None: flux_bins = range(n_flux_bins)
    if background_bins is None: background_bins = range(n_background_bins)

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
                    model_trail.append(line.model_full_trail_untrailed)
                    model_noise.append(line.model_full_trail_noise)
                    model_untrailed.append(line.model_full_trail)
                    n_lines_used += 1

    if n_lines_used == 0: return None, None, None
    print("n_lines_used",n_lines_used)
    print(len(model_trail))

    # Duplicate the x arrays for all trails
    # x_all = [np.arange(ut.trail_length) + 1] * n_lines_used

    return model_trail, model_noise, model_untrailed


class CTImodel:

    def __init__(
            self,
            trap_density_A=1.0,
            trap_release_time_A=0.0,
            warm_pixel_flux=1.0,
    ):
        self.trap_density_A = trap_density_A
        self.trap_release_time_A = trap_release_time_A
        self.warm_pixel_flux = warm_pixel_flux

    """
    An instance of this class will be available during model fitting.
    This method will be used to fit the model to data and compute a likelihood.
    """

    def trail_from_xvalues(self, xvalues):

        # date = 2400000.5 + image_A.header.modified_julian_date
        # roe, ccd, traps = ac.CTI_model_for_HST_ACS(date)
        traps = [
            arcticpy.TrapInstantCapture(
                density=self.trap_density_A,
                release_timescale=self.trap_release_time_A
            )
        ]
        roe = arcticpy.ROE()
        ccd = arcticpy.CCD(full_well_depth=84700, well_fill_power=0.478)
        model_after_trail = []

        # number of CCD columns to model
        for eper in xvalues:
            # xvalues are [length_of_trail,background,hot_pixel_number]
            model_before_trail = np.full(eper[0], eper[1])
            model_before_trail[eper[2]] = self.warm_pixel_flux
            print("model before trail",model_before_trail[-15:])

            # Run arCTIc to produce a model of the trailed CCD column
            model_after_trail.append(
                arcticpy.add_cti(
                    model_before_trail,
                    parallel_roe=roe,
                    parallel_ccd=ccd,
                    parallel_traps=traps,
                    parallel_express=5,
                    verbosity=1
                )
            )

        return model_after_trail

class Analysis(af.Analysis):

    def __init__(self, data, noise_map):
        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        The 'instance' that comes into this method is an instance of the Gaussian class
        above, with the parameters set to values chosen by the non-linear search.
        """

        print("CTImodel Instance:")
        print("trap_density_A = ", instance.trap_density_A)
        print("trap_release_time_A = ", instance.trap_release_time_A)
        print("warm_pixel_flux = ", instance.warm_pixel_flux)

        if instance.trap_density_A < 0: return -np.inf
        if instance.trap_release_time_A < 0: return -np.inf
        if instance.warm_pixel_flux < 0: return -np.inf

        """
        We fit the ``data`` with the Gaussian instance, using its
        "profile_from_xvalues" function to create the model data.
        """
        #print(self.data)
        print(len(self.data))
        #print(self.data[0])
        xvalues = []
        for eper in self.data:
            print('len(eper)',len(eper),np.argmax(eper),len(eper)-np.argmax(eper),eper.shape)
            xvalues.append( [ len(eper), eper[0], np.argmax(eper) ] )


        print(len(xvalues))

        model_data = instance.trail_from_xvalues(xvalues=xvalues)

        chi_squareds = []
        for eper in self.data:
            chi_squareds.append( np.sum( model_data[0] / self.noise_map ) )
        #chi_squared = sum(chi_squareds)
        #residual_map = self.data - model_data
        #chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squareds)

        return log_likelihood
