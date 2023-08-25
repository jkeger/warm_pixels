import logging
from pathlib import Path
import warm_pixels as wp
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import autofit as af
from warm_pixels import hst_utilities as ut#, PixelLine
from warm_pixels import misc
#from warm_pixels.hst_functions.fit import fit_dataset_total_trap_density
#from warm_pixels.hst_functions.trail_model import trail_model_hst
from warm_pixels.hst_functions.trail_model_k_fastest import trail_model_arctic_notch_pushed 
from warm_pixels.hst_functions.trail_model_k_fastest import trail_model_arctic_notch_pushed_plot
#from warm_pixels.fit.model import TrailModel
#from warm_pixels.fit.analysis import Analysis
from warm_pixels.model.group import QuadrantGroup
from autoarray.fit.fit_dataset import SimpleFit
import time
import csv
from os import path
import sys
import os
import pathlib
import shutil
from astropy.io import fits
import copy

from autoarray.structures.header import Header
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.layout.layout import Layout2D
from autoarray.layout.region import Region2D

from autoarray import exc
from autoarray.structures.arrays import array_2d_util
from autoarray.layout import layout_util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")

cosma_id = int(sys.argv[1])
const_fix = float(sys.argv[2])
rho_fix = 1
dataset_date = str(sys.argv[3])

logger = logging.getLogger(
    __name__
)

start_time = time.time()
# Sum of exponentials model to fit negative rho_q after correction
def trail_model_exp(x, rho_q, n_e, n_bg, row, beta, w, A, B, C, tau_a, tau_b, tau_c, notch):
    
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
    
    
    #print('first term denominator =', (w - notch))
    local_counter=0
    local_array=[]
    
    #print('len n_bg is', len(n_bg))
    while local_counter<len(n_bg):
# =============================================================================
#         term1=np.abs(n_e[local_counter]) - notch
#         #print('term1 is', term1)
#         term2=np.abs(n_bg[local_counter]) - notch
#         #print('term2 is', term2)
# =============================================================================
        volume1 = np.sign(n_e[local_counter]) * np.clip((abs(n_e[local_counter]) - notch) / (w - notch), 0, 1) ** beta
        volume2 = np.sign(n_bg[local_counter]) * np.clip((abs(n_bg[local_counter]) - notch) / (w - notch), 0, 1) ** beta
        local_array.append(
                rho_q
        * (volume1 - volume2)
        * row[local_counter]
        * (
                A * np.exp((1 - x[local_counter]) / tau_a) * (1 - np.exp(-1 / tau_a))
                + B * np.exp((1 - x[local_counter]) / tau_b) * (1 - np.exp(-1 / tau_b))
                + C * np.exp((1 - x[local_counter]) / tau_c) * (1 - np.exp(-1 / tau_c))
        )  
        )
        local_counter=local_counter+1
    return (local_array)
# Define classes 
class Analysis(af.Analysis):
        def __init__(self, x, y, noise, generated_trails):
            self.x = x
            self.y = y
            self.noise = noise
            self.generated_trails = generated_trails
    
        def visualize(self, paths, instance, during_analysis):
            #plt.plot(self.x, self.y)
            #plt.plot(self.x, instance(
            #    x=self.x,
            #    n_e=self.n_e,
            #    n_bg=self.n_bg,
            #    row=self.row,
            #)
            #)
            print('Visualising')
    
        def log_likelihood_function(self, instance):
            modelled_trail = instance(
                x=self.x,
                generated_trails=self.generated_trails
            )
            fit = SimpleFit(
                data=self.y,
                model_data=modelled_trail,
                noise_map=self.noise,
            )
            print('log likelihood = ', fit.log_likelihood)
            return fit.log_likelihood
        
class Analysis2(af.Analysis):
        def __init__(self, x, y, noise, n_e, n_bg, row):
            self.x = x
            self.y = y
            self.noise = noise
            self.n_e = n_e
            self.n_bg = n_bg
            self.row = row
    
        def visualize(self, paths, instance, during_analysis):
            #plt.plot(self.x, self.y)
            #plt.plot(self.x, instance(
            #    x=self.x,
            #    n_e=self.n_e,
            #    n_bg=self.n_bg,
            #    row=self.row,
            #)
            #)
            print('Visualising')
    
        def log_likelihood_function(self, instance):
            modelled_trail = instance(
                x=self.x,
                n_e=self.n_e,
                n_bg=self.n_bg,
                row=self.row,
            )
            fit = SimpleFit(
                data=self.y,
                model_data=modelled_trail,
                noise_map=self.noise,
            )
            #print('x =', self.x)
            #print('modelled_trail= ', modelled_trail)
            #print('log likelihood = ', fit.log_likelihood)        
            return fit.log_likelihood
class TrailModel:
        def __init__(
                self,
                days_var,
                rho_q,
                beta,
                w,
                a,
                b,
                c,
                tau_a,
                tau_b,
                tau_c,
                notch
        ):
            self.rho_q = rho_q
            self.days_var=days_var
            self.beta = beta
            self.w = w
            self.a = a
            self.b = b
            self.c = c
            self.tau_a = tau_a
            self.tau_b = tau_b
            self.tau_c = tau_c
            self.notch=notch
    
        def __call__(self, x, generated_trails):
# =============================================================================
#             print('rho_q=',self.rho_q)
#             print('beta=',self.beta)
#             print('a=',self.a)
#             print('b=',self.b)
#             print('c=',self.c)
#             print('tau_a=',self.tau_a)
#             print('tau_b=',self.tau_b)
#             print('tau_c=',self.tau_c)
#             print('notch=',self.notch)
# =============================================================================
            return trail_model_arctic_notch_pushed(
                x=x,
                rho_q=self.rho_q,
                generated_trails = generated_trails,
                beta=self.beta,
                w=self.w,
                A=self.a,
                B=self.b,
                C=self.c,
                tau_a=self.tau_a,
                tau_b=self.tau_b,
                tau_c=self.tau_c,
                notch=self.notch,
            )
class TrailModelPrint:
        def __init__(
                self,
                days_var,
                rho_q,
                beta,
                w,
                a,
                b,
                c,
                tau_a,
                tau_b,
                tau_c,
                notch
        ):
            self.rho_q = rho_q
            self.days_var=days_var
            self.beta = beta
            self.w = w
            self.a = a
            self.b = b
            self.c = c
            self.tau_a = tau_a
            self.tau_b = tau_b
            self.tau_c = tau_c
            self.notch=notch
    
        def __call__(self, x, n_e, n_bg, row):
# =============================================================================
#             print('rho_q=',self.rho_q)
#             print('beta=',self.beta)
#             print('a=',self.a)
#             print('b=',self.b)
#             print('c=',self.c)
#             print('tau_a=',self.tau_a)
#             print('tau_b=',self.tau_b)
#             print('tau_c=',self.tau_c)
#             print('notch=',self.notch)
#             print('trail value=', trail_model_exp(
#                 x=x,
#                 rho_q=self.rho_q,
#                 n_e=n_e,
#                 n_bg=n_bg,
#                 row=row,
#                 beta=self.beta,
#                 w=self.w,
#                 A=self.a,
#                 B=self.b,
#                 C=self.c,
#                 tau_a=self.tau_a,
#                 tau_b=self.tau_b,
#                 tau_c=self.tau_c,
#                 notch=self.notch,
#             ) )
# =============================================================================
            return trail_model_exp(
                x=x,
                rho_q=self.rho_q,
                n_e=n_e,
                n_bg=n_bg,
                row=row,
                beta=self.beta,
                w=self.w,
                A=self.a,
                B=self.b,
                C=self.c,
                tau_a=self.tau_a,
                tau_b=self.tau_b,
                tau_c=self.tau_c,
                notch=self.notch,
            )
        
def fits_hdu_via_quadrant_letter_from(quadrant_letter):

    if quadrant_letter == "D" or quadrant_letter == "C":
        return 1
    elif quadrant_letter == "B" or quadrant_letter == "A":
        return 4
    else:
        raise exc.ArrayException("Quadrant letter for FrameACS must be A, B, C or D.")


def array_eps_to_counts(array_eps, bscale, bzero):

    if bscale is None:
        raise exc.ArrayException(
            "Cannot convert a Frame2D to units COUNTS without a bscale attribute (bscale = None)."
        )

    return (array_eps - bzero) / bscale


class Array2DACS(Array2D):
    """
    An ACS array consists of four quadrants ('A', 'B', 'C', 'D') which have the following layout (which are described
    at the following STScI 
    link https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418).

       <--------S-----------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []          /\
    /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /        |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
    P   [xxxxxxxxx B/C xxxxxxx] [xxxxxxxxx A/D xxxxxxx]  P         | clocks an image
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
    \/  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  \/        | (e.g. towards row 0
                                                                   | of the NumPy arrays)

    For a ACS .fits file:

    - The images contained in hdu 1 correspond to quadrants B (left) and A (right).
    - The images contained in hdu 4 correspond to quadrants C (left) and D (right).
    """

    @classmethod
    def from_fits(cls, file_path, quadrant_letter):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.
        
        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        hdu = fits_hdu_via_quadrant_letter_from(quadrant_letter=quadrant_letter)

        array = array_2d_util.numpy_array_2d_via_fits_from(file_path=file_path, hdu=hdu)

        return cls.from_ccd(array_electrons=array, quadrant_letter=quadrant_letter)

    @classmethod
    def from_ccd(
        cls,
        array_electrons,
        quadrant_letter,
        header=None,
        bias_subtract_via_prescan=False,
        bias=None,
    ):
        """
        Using an input array of both quadrants in electrons, use the quadrant letter to extract the quadrant from the
        full CCD and perform the rotations required to give correct arctic.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.
        
        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """
        if quadrant_letter == "A":

            array_electrons = array_electrons[0:2068, 0:2072]
            roe_corner = (1, 0)
            use_flipud = True

            if bias is not None:
                bias = bias[0:2068, 0:2072]

        elif quadrant_letter == "B":

            array_electrons = array_electrons[0:2068, 2072:4144]
            roe_corner = (1, 1)
            use_flipud = True

            if bias is not None:
                bias = bias[0:2068, 2072:4144]

        elif quadrant_letter == "C":

            array_electrons = array_electrons[0:2068, 0:2072]

            roe_corner = (1, 0)
            use_flipud = False

            if bias is not None:
                bias = bias[0:2068, 0:2072]

        elif quadrant_letter == "D":

            array_electrons = array_electrons[0:2068, 2072:4144]

            roe_corner = (1, 1)
            use_flipud = False

            if bias is not None:
                bias = bias[0:2068, 2072:4144]

        else:
            raise exc.ArrayException(
                "Quadrant letter for FrameACS must be A, B, C or D."
            )

        return cls.quadrant_a(
            array_electrons=array_electrons,
            header=header,
            roe_corner=roe_corner,
            use_flipud=use_flipud,
            bias_subtract_via_prescan=bias_subtract_via_prescan,
            bias=bias,
        )

    @classmethod
    def quadrant_a(
        cls,
        array_electrons,
        roe_corner,
        use_flipud,
        header=None,
        bias_subtract_via_prescan=False,
        bias=None,
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.
        
        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=roe_corner
        )

        if use_flipud:
            array_electrons = np.flipud(array_electrons)

        if bias_subtract_via_prescan:

            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=roe_corner
            )

            if use_flipud:
                bias = np.flipud(bias)

            array_electrons -= bias

            header.bias = Array2DACS.manual_native(array=bias, pixel_scales=0.05)

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_b(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 1)
        )

        array_electrons = np.flipud(array_electrons)

        if bias_subtract_via_prescan:

            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=(1, 1)
            )

            bias = np.flipud(bias)

            array_electrons -= bias

            header.bias = Array2DACS.manual_native(array=bias, pixel_scales=0.05)

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_c(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 0)
        )

        if bias_subtract_via_prescan:

            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=(1, 0)
            )

            array_electrons -= bias

            header.bias = Array2DACS.manual_native(array=bias, pixel_scales=0.05)

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_d(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 1)
        )

        if bias_subtract_via_prescan:
            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=(1, 1)
            )

            array_electrons -= bias

            header.bias = Array2DACS.manual_native(array=bias, pixel_scales=0.05)

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    def update_fits(self, original_file_path, new_file_path):
        """
        Output the array to a .fits file.

        Parameters
        ----------
        file_path
            The path the file is output to, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        """

        new_file_dir = os.path.split(new_file_path)[0]

        if not os.path.exists(new_file_dir):

            os.makedirs(new_file_dir)

        if not os.path.exists(new_file_path):

            shutil.copy(original_file_path, new_file_path)

        hdulist = fits.open(new_file_path)

        hdulist[self.header.hdu].data = self.layout_2d.original_orientation_from(
            array=self
        )

        ext_header = hdulist[4].header
        bscale = ext_header["BSCALE"]

        os.remove(new_file_path)

        hdulist.writeto(new_file_path)


class ImageACS(Array2DACS):
    """
    The layout of an ACS array and image is given in `FrameACS`.

    This class handles specifically the image of an ACS observation, assuming that it contains specific
    header info.
    """

    @classmethod
    def get_MJD(
        cls,
        file_path,
        quadrant_letter,
    ):
        hdu = fits_hdu_via_quadrant_letter_from(quadrant_letter=quadrant_letter)

        header_sci_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=0)
        header_hdu_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=hdu)
        
        
        header = HeaderACS(
            header_sci_obj=header_sci_obj,
            header_hdu_obj=header_hdu_obj,
            hdu=hdu,
            quadrant_letter=quadrant_letter,
        )
        
        global MJD_var
        MJD_var = header.MJD
        print('MJD_var is', MJD_var)
        
        
    @classmethod
    def from_fits(
        cls,
        file_path,
        quadrant_letter,
        bias_subtract_via_bias_file=False,
        bias_subtract_via_prescan=False,
        bias_file_path=None,
        use_calibrated_gain=True,
    ):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418

        Parameters
        ----------
        file_path
            The full path of the file that the image is loaded from, including the file name and ``.fits`` extension.
        quadrant_letter
            The letter of the ACS quadrant the image is extracted from and loaded.
        bias_subtract_via_bias_file
            If True, the corresponding bias file of the image is loaded (via the name of the file in the fits header).
        bias_subtract_via_prescan
            If True, the prescan on the image is used to estimate a component of bias that is subtracted from the image.
        bias_file_path
            If `bias_subtract_via_bias_file=True`, this overwrites the path to the bias file instead of the default
            behaviour of using the .fits header.
        use_calibrated_gain
            If True, the calibrated gain values are used to convert from COUNTS to ELECTRONS.
        """
        
        
        hdu = fits_hdu_via_quadrant_letter_from(quadrant_letter=quadrant_letter)

        header_sci_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=0)
        header_hdu_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=hdu)
        
        
        header = HeaderACS(
            header_sci_obj=header_sci_obj,
            header_hdu_obj=header_hdu_obj,
            hdu=hdu,
            quadrant_letter=quadrant_letter,
        )
        
        global MJD_var
        MJD_var = header.MJD
        print('MJD_var is', MJD_var)
        
        global CCDGAIN_var
        CCDGAIN_var = header.gain
        print('CCDGAIN_var is', CCDGAIN_var)
        
        
        

        if header.header_sci_obj["TELESCOP"] != "HST":
            raise exc.ArrayException(
                f"The file {file_path} does not point to a valid HST ACS dataset."
            )

        if header.header_sci_obj["INSTRUME"] != "ACS":
            raise exc.ArrayException(
                f"The file {file_path} does not point to a valid HST ACS dataset."
            )

        array = array_2d_util.numpy_array_2d_via_fits_from(
            file_path=file_path, hdu=hdu, do_not_scale_image_data=True
        )

        array = header.array_original_to_electrons(
            array=array, use_calibrated_gain=use_calibrated_gain
        )
        
        
        if bias_subtract_via_bias_file:

            if bias_file_path is None:

                file_dir = os.path.split(file_path)[0]
                bias_file_path = path.join(file_dir, header.bias_file)
                
            if bias_file_path is not None: # Added functionality to point to new directory - Paolo
                file_dir = Path(bias_file_path)
                bias_file_path = path.join(file_dir, header.bias_file)

            bias = array_2d_util.numpy_array_2d_via_fits_from(
                file_path=bias_file_path, hdu=hdu, do_not_scale_image_data=True
            )

            header_sci_obj = array_2d_util.header_obj_from(
                file_path=bias_file_path, hdu=0
            )
            header_hdu_obj = array_2d_util.header_obj_from(
                file_path=bias_file_path, hdu=hdu
            )

            bias_header = HeaderACS(
                header_sci_obj=header_sci_obj,
                header_hdu_obj=header_hdu_obj,
                hdu=hdu,
                quadrant_letter=quadrant_letter,
            )

            if bias_header.original_units != "COUNTS":

                raise exc.ArrayException("Cannot use bias frame not in counts.")
            
            
                

            bias = bias * bias_header.calibrated_gain

        else:

            bias = None

        return cls.from_ccd(
            array_electrons=array,
            quadrant_letter=quadrant_letter,
            header=header,
            bias_subtract_via_prescan=bias_subtract_via_prescan,
            bias=bias,
        )


class paolo_ImageACS(Array2DACS):
    """
    The layout of an ACS array and image is given in `FrameACS`.

    This class handles specifically the image of an ACS observation, assuming that it contains specific
    header info.
    """

    @classmethod
    def from_fits(
        cls,
        file_path,
        quadrant_letter,
        bias_subtract_via_bias_file=False,
        bias_subtract_via_prescan=False,
        bias_file_path=None,
        use_calibrated_gain=True,
    ):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418

        Parameters
        ----------
        file_path
            The full path of the file that the image is loaded from, including the file name and ``.fits`` extension.
        quadrant_letter
            The letter of the ACS quadrant the image is extracted from and loaded.
        bias_subtract_via_bias_file
            If True, the corresponding bias file of the image is loaded (via the name of the file in the fits header).
        bias_subtract_via_prescan
            If True, the prescan on the image is used to estimate a component of bias that is subtracted from the image.
        bias_file_path
            If `bias_subtract_via_bias_file=True`, this overwrites the path to the bias file instead of the default
            behaviour of using the .fits header.
        use_calibrated_gain
            If True, the calibrated gain values are used to convert from COUNTS to ELECTRONS.
        """
        
        
        hdu = fits_hdu_via_quadrant_letter_from(quadrant_letter=quadrant_letter)

        header_sci_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=0)
        header_hdu_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=hdu)

        header = HeaderACS(
            header_sci_obj=header_sci_obj,
            header_hdu_obj=header_hdu_obj,
            hdu=hdu,
            quadrant_letter=quadrant_letter,
        )

        if header.header_sci_obj["TELESCOP"] != "HST":
            raise exc.ArrayException(
                f"The file {file_path} does not point to a valid HST ACS dataset."
            )

        if header.header_sci_obj["INSTRUME"] != "ACS":
            raise exc.ArrayException(
                f"The file {file_path} does not point to a valid HST ACS dataset."
            )

        array = array_2d_util.numpy_array_2d_via_fits_from(
            file_path=file_path, hdu=hdu, do_not_scale_image_data=True
        )

        array = header.paolo_array_original_to_electrons(
            array=array, use_calibrated_gain=use_calibrated_gain
        )
        
        
        
        
        if bias_subtract_via_bias_file:

            if bias_file_path is None:

                file_dir = os.path.split(file_path)[0]
                bias_file_path = path.join(file_dir, header.bias_file)

            bias = array_2d_util.numpy_array_2d_via_fits_from(
                file_path=bias_file_path, hdu=hdu, do_not_scale_image_data=True
            )

            header_sci_obj = array_2d_util.header_obj_from(
                file_path=bias_file_path, hdu=0
            )
            header_hdu_obj = array_2d_util.header_obj_from(
                file_path=bias_file_path, hdu=hdu
            )

            bias_header = HeaderACS(
                header_sci_obj=header_sci_obj,
                header_hdu_obj=header_hdu_obj,
                hdu=hdu,
                quadrant_letter=quadrant_letter,
            )

            if bias_header.original_units != "COUNTS":

                raise exc.ArrayException("Cannot use bias frame not in counts.")

            bias = bias * bias_header.calibrated_gain

        else:

            bias = None

        return cls.from_ccd(
            array_electrons=array,
            quadrant_letter=quadrant_letter,
            header=header,
            bias_subtract_via_prescan=bias_subtract_via_prescan,
            bias=bias,
        )
class Layout2DACS(Layout2D):
    @classmethod
    def from_sizes(cls, roe_corner, serial_prescan_size=24, parallel_overscan_size=20):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.
        """

        parallel_overscan = Region2D(
            (2068 - parallel_overscan_size, 2068, serial_prescan_size, 2072)
        )

        serial_prescan = Region2D((0, 2068, 0, serial_prescan_size))

        return Layout2D.rotated_from_roe_corner(
            roe_corner=roe_corner,
            shape_native=(2068, 2072),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
        )


class HeaderACS(Header):
    def __init__(
        self,
        header_sci_obj,
        header_hdu_obj,
        quadrant_letter=None,
        hdu=None,
        bias=None,
        bias_serial_prescan_column=None,
    ):

        super().__init__(header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj)

        self.bias = bias
        self.bias_serial_prescan_column = bias_serial_prescan_column
        self.quadrant_letter = quadrant_letter
        self.hdu = hdu

    @property
    def bscale(self):
        return self.header_hdu_obj["BSCALE"]

    @property
    def bzero(self):
        return self.header_hdu_obj["BZERO"]

    @property
    def gain(self):
        return self.header_sci_obj["CCDGAIN"]

    @property
    def calibrated_gain(self):

        if round(self.gain) == 1:
            calibrated_gain = [0.99989998, 0.97210002, 1.01070000, 1.01800000]
        elif round(self.gain) == 2:
            calibrated_gain = [2.002, 1.945, 2.028, 1.994]
        elif round(self.gain) == 4:
            calibrated_gain = [4.011, 3.902, 4.074, 3.996]
        else:
            raise exc.ArrayException(
                "Calibrated gain of ACS file does not round to 1, 2 or 4."
            )

        if self.quadrant_letter == "A":
            return calibrated_gain[0]
        elif self.quadrant_letter == "B":
            return calibrated_gain[1]
        elif self.quadrant_letter == "C":
            return calibrated_gain[2]
        elif self.quadrant_letter == "D":
            return calibrated_gain[3]

    @property
    def original_units(self):
        return self.header_hdu_obj["BUNIT"]

    @property
    def bias_file(self):
        return self.header_sci_obj["BIASFILE"].replace("jref$", "")
    
    @property
    def MJD(self):
        return self.modified_julian_date
    
    def array_eps_to_counts(self, array_eps):
        return array_eps_to_counts(
            array_eps=array_eps, bscale=self.bscale, bzero=self.bzero
        )

    def array_original_to_electrons(self, array, use_calibrated_gain):

        if self.original_units in "COUNTS":
            global paolo_bscale
            global paolo_bzero
            #paolo_bscale = self.bscale
            #paolo_bzero = self.bzero
            paolo_bscale = 1
            paolo_bzero = 0
            array = (array * self.bscale) + self.bzero
            
        elif self.original_units in "CPS":
            array = (array * self.exposure_time * self.bscale) + self.bzero

        if use_calibrated_gain:
            return array * self.calibrated_gain
        else:
            return array * self.gain
        
    def paolo_array_original_to_electrons(self, array, use_calibrated_gain):

        if self.original_units in "COUNTS":
            array = (array * 1) + 0
            
        elif self.original_units in "CPS":
            array = (array * self.exposure_time * self.bscale) + self.bzero

        if use_calibrated_gain:
            return array * self.calibrated_gain
        else:
            return array * self.gain
    
    
    

    def array_electrons_to_original(self, array, use_calibrated_gain):

        if use_calibrated_gain:
            array /= self.calibrated_gain
        else:
            array /= self.gain

        if self.original_units in "COUNTS":
            return (array - self.bzero) / self.bscale
        elif self.original_units in "CPS":
            return (array - self.bzero) / (self.exposure_time * self.bscale)
        
    def paolo_array_electrons_to_original(self, array, use_calibrated_gain):

        if use_calibrated_gain:
            array /= self.calibrated_gain
        else:
            array /= self.gain

        if self.original_units in "COUNTS":
            return (array - 0) / 1
        elif self.original_units in "CPS":
            return (array - 0) / (self.exposure_time * 1)


def prescan_fitted_bias_column(prescan, n_rows=2048, n_rows_ov=20):
    """
    Generate a bias column to be subtracted from the main image by doing a
    least squares fit to the serial prescan region.

    e.g. image -= prescan_fitted_bias_column(image[18:24])

    See Anton & Rorres (2013), S9.3, p460.

    Parameters
    ----------
    prescan : [[float]]
        The serial prescan part of the image. Should usually cover the full
        number of rows but may skip the first few columns of the prescan to
        avoid trails.

    n_rows
        The number of rows in the image, exculding overscan.

    n_rows_ov, int
        The number of overscan rows in the image.

    Returns
    -------
    bias_column : [float]
        The fitted bias to be subtracted from all image columns.
    """
    n_columns_fit = prescan.shape[1]

    # Flatten the multiple fitting columns to a long 1D array
    # y = [y_1_1, y_2_1, ..., y_nrow_1, y_1_2, y_2_2, ..., y_nrow_ncolfit]
    y = prescan[:-n_rows_ov].T.flatten()
    # x = [1, 2, ..., nrow, 1, ..., nrow, 1, ..., nrow, ...]
    x = np.tile(np.arange(n_rows), n_columns_fit)

    # M = [[1, 1, ..., 1], [x_1, x_2, ..., x_n]].T
    M = np.array([np.ones(n_rows * n_columns_fit), x]).T

    # Best-fit values for y = M v
    v = np.dot(np.linalg.inv(np.dot(M.T, M)), np.dot(M.T, y))

    # Map to full image size for easy subtraction
    bias_column = v[0] + v[1] * np.arange(n_rows + n_rows_ov)

    return np.transpose([bias_column])


def output_quadrants_to_fits(
    file_path: str,
    quadrant_a,
    quadrant_b,
    quadrant_c,
    quadrant_d,
    header_a=None,
    header_b=None,
    header_c=None,
    header_d=None,
    overwrite: bool = False,
):

    file_dir = os.path.split(file_path)[0]

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    array_hdu_1 = np.zeros((2068, 4144))
    array_hdu_4 = np.zeros((2068, 4144))

    def get_header(quadrant):
        try:
            return quadrant.header
        except AttributeError:
            raise (
                "You must pass in the header of the quadrants to output them to an ACS fits file."
            )

    header_a = get_header(quadrant_a) if header_a is None else header_a
    try:
        quadrant_a = copy.copy(np.asarray(quadrant_a.native))
    except AttributeError:
        quadrant_a = copy.copy(np.asarray(quadrant_a))

    quadrant_a = quadrant_convert_to_original(
        quadrant=quadrant_a, roe_corner=(1, 0), header=header_a, use_flipud=True
    )
    array_hdu_4[0:2068, 0:2072] = quadrant_a

    header_b = get_header(quadrant_b) if header_b is None else header_b

    try:
        quadrant_b = copy.copy(np.asarray(quadrant_b.native))
    except AttributeError:
        quadrant_b = copy.copy(np.asarray(quadrant_b))
    quadrant_b = quadrant_convert_to_original(
        quadrant=quadrant_b, roe_corner=(1, 1), header=header_b, use_flipud=True
    )
    array_hdu_4[0:2068, 2072:4144] = quadrant_b

    header_c = get_header(quadrant_c) if header_c is None else header_c
    try:
        quadrant_c = copy.copy(np.asarray(quadrant_c.native))
    except AttributeError:
        quadrant_c = copy.copy(np.asarray(quadrant_c))
    quadrant_c = quadrant_convert_to_original(
        quadrant=quadrant_c, roe_corner=(1, 0), header=header_c, use_flipud=False
    )
    array_hdu_1[0:2068, 0:2072] = quadrant_c

    header_d = get_header(quadrant_d) if header_d is None else header_d
    try:
        quadrant_d = copy.copy(np.asarray(quadrant_d.native))
    except AttributeError:
        quadrant_d = copy.copy(np.asarray(quadrant_d))
    quadrant_d = quadrant_convert_to_original(
        quadrant=quadrant_d, roe_corner=(1, 1), header=header_d, use_flipud=False
    )
    array_hdu_1[0:2068, 2072:4144] = quadrant_d

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU(array_hdu_1))
    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU(array_hdu_4))
    hdu_list.append(fits.ImageHDU())

    def set_header(header):
        header.set("cticor", "ARCTIC", "CTI CORRECTION PERFORMED USING ARCTIC")
        return header

    hdu_list[0].header = set_header(header_a.header_sci_obj)
    hdu_list[1].header = set_header(header_c.header_hdu_obj)
    hdu_list[4].header = set_header(header_a.header_hdu_obj)
    hdu_list.writeto(file_path)

def paolo_output_quadrants_to_fits(
    file_path: str,
    quadrant_a,
    quadrant_b,
    quadrant_c,
    quadrant_d,
    header_a=None,
    header_b=None,
    header_c=None,
    header_d=None,
    overwrite: bool = False,
):

    file_dir = os.path.split(file_path)[0]

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    array_hdu_1 = np.zeros((2068, 4144))
    array_hdu_4 = np.zeros((2068, 4144))

    def get_header(quadrant):
        try:
            return quadrant.header
        except AttributeError:
            raise (
                "You must pass in the header of the quadrants to output them to an ACS fits file."
            )

    header_a = get_header(quadrant_a) if header_a is None else header_a
    try:
        quadrant_a = copy.copy(np.asarray(quadrant_a.native))
    except AttributeError:
        quadrant_a = copy.copy(np.asarray(quadrant_a))

    quadrant_a = paolo_quadrant_convert_to_original(
        quadrant=quadrant_a, roe_corner=(1, 0), header=header_a, use_flipud=True
    )
    array_hdu_4[0:2068, 0:2072] = quadrant_a

    header_b = get_header(quadrant_b) if header_b is None else header_b

    try:
        quadrant_b = copy.copy(np.asarray(quadrant_b.native))
    except AttributeError:
        quadrant_b = copy.copy(np.asarray(quadrant_b))
    quadrant_b = paolo_quadrant_convert_to_original(
        quadrant=quadrant_b, roe_corner=(1, 1), header=header_b, use_flipud=True
    )
    array_hdu_4[0:2068, 2072:4144] = quadrant_b

    header_c = get_header(quadrant_c) if header_c is None else header_c
    try:
        quadrant_c = copy.copy(np.asarray(quadrant_c.native))
    except AttributeError:
        quadrant_c = copy.copy(np.asarray(quadrant_c))
    quadrant_c = paolo_quadrant_convert_to_original(
        quadrant=quadrant_c, roe_corner=(1, 0), header=header_c, use_flipud=False
    )
    array_hdu_1[0:2068, 0:2072] = quadrant_c

    header_d = get_header(quadrant_d) if header_d is None else header_d
    try:
        quadrant_d = copy.copy(np.asarray(quadrant_d.native))
    except AttributeError:
        quadrant_d = copy.copy(np.asarray(quadrant_d))
    quadrant_d = paolo_quadrant_convert_to_original(
        quadrant=quadrant_d, roe_corner=(1, 1), header=header_d, use_flipud=False
    )
    array_hdu_1[0:2068, 2072:4144] = quadrant_d

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU(array_hdu_1))
    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU(array_hdu_4))
    hdu_list.append(fits.ImageHDU())

    def set_header(header):
        header.set("cticor", "ARCTIC", "CTI CORRECTION PERFORMED USING ARCTIC")
        return header

    hdu_list[0].header = set_header(header_a.header_sci_obj)
    hdu_list[1].header = set_header(header_c.header_hdu_obj)
    hdu_list[4].header = set_header(header_a.header_hdu_obj)
    hdu_list.writeto(file_path)

def quadrant_convert_to_original(
    quadrant, roe_corner, header, use_flipud=False, use_calibrated_gain=True
):

    if header.bias is not None:
        quadrant += header.bias.native

    if header.bias_serial_prescan_column is not None:
        quadrant += header.bias_serial_prescan_column

    quadrant = header.array_electrons_to_original(
        array=quadrant, use_calibrated_gain=use_calibrated_gain
    )

    if use_flipud:
        quadrant = np.flipud(quadrant)

    return layout_util.rotate_array_via_roe_corner_from(
        array=quadrant, roe_corner=roe_corner
    )

def paolo_quadrant_convert_to_original(
    quadrant, roe_corner, header, use_flipud=False, use_calibrated_gain=True
):

    if header.bias is not None:
        quadrant += header.bias.native

    if header.bias_serial_prescan_column is not None:
        quadrant += header.bias_serial_prescan_column

    quadrant = header.paolo_array_electrons_to_original(
        array=quadrant, use_calibrated_gain=use_calibrated_gain
    )

    if use_flipud:
        quadrant = np.flipud(quadrant)

    return layout_util.rotate_array_via_roe_corner_from(
        array=quadrant, roe_corner=roe_corner
    )
        
 # Define a function for the 50 plot diagram before the fitting. This includes calculating the best fit arctic model.
def Paolo_autofit_global_50(group: QuadrantGroup, use_corrected=False, save_path=None): 
    
    stacked_lines = group.stacked_lines()
    
    
    
    # Define constants and free variables
    # CCD
    beta = 0.556
    w = 84700.0
    # Trap species
    a = 0.180
    b = 0.789
    c = 0.032
    # Trap lifetimes before or after the temperature change
# =============================================================================
#     if date < ut.date_T_change:
#         tau_a = 0.48
#         tau_b = 4.86
#         tau_c = 20.6
#     else:
#         tau_a = 0.74
#         tau_b = 7.70
#         tau_c = 37.0
#     
# =============================================================================
    tau_a = 0.74
    tau_b = 7.70
    tau_c = 37.0
    notch=0.0
    
    # Convert MJD to days since launch for notch time evolution
    global days_var
    JD_var=float(MJD_var)+2400000.5
    days_var=JD_var-2452334.5
    #notch=0.013468157265719103*days_var-0.13793219313191085
    notch=96.33892681649918
    
    if days_var < 1586.5:
        tau_a=0.354
        tau_b=4.082
        tau_c=31.8176
    else:
        tau_a=0.541
        tau_b=6.466
        tau_c=57.004
    

    
    # CCD
    rho_q = af.UniformPrior(
        lower_limit=-10.0,
        #lower_limit=0.0,
        upper_limit=10.0,
    )
  
    beta = af.GaussianPrior(
              mean=0.556,
              sigma=0.1,
       )
    
    # Trap species
    a = af.UniformPrior(
        lower_limit=0,
        upper_limit=1.0,
    )
    b = af.UniformPrior(
        lower_limit=0,
        upper_limit=1.0,
    )
    c = 1 - (a + b)
    
    tau_a = af.GaussianPrior(
          mean=tau_a,
          sigma=0.2,
      )
    tau_b = af.GaussianPrior(
          mean=tau_b,
          sigma=2.0,
      )
    tau_c = af.GaussianPrior(
          mean=tau_c,
          sigma=10.0,
      )
    notch = af.UniformPrior(
        lower_limit=-10000.0,
        #lower_limit=0.0,
        upper_limit=10000.0,
    )
    
    model = af.Model(
        TrailModel,
        days_var=days_var,
        rho_q=rho_q,
        beta=beta,
        w=w,
        a=a,
        b=b,
        c=c,
        tau_a=tau_a,
        tau_b=tau_b,
        tau_c=tau_c,
        notch=notch,
    )
    
    model.add_assertion(c > 0.0)
    model.add_assertion(tau_a > 0.0)
    model.add_assertion(tau_b > 0.0)
    model.add_assertion(tau_c > 0.0)
   
    
    # Extract row bins
    n_row_bins = stacked_lines.n_row_bins
    n_flux_bins = stacked_lines.n_flux_bins
    n_background_bins = stacked_lines.n_background_bins

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
    fontsize = 20

    
    # Compile the data from all stacked lines for the global AUTOFIT
    n_lines_used = 0
    y_all = np.array([])
    #x_all = np.array([])
    noise_all_temp = np.array([])
    noise_all = np.array([])
    generated_trails=[]
    
   #x_one=np.array(np.arange(ut.trail_length)+1)

                        
    # Find the noise, bg, row for each subplot
    for i_row in range(n_row_bins):
        for i_flux in range(n_flux_bins):
            ax = axes[n_row_bins - 1 - i_row][i_flux]
            for i_background, c in enumerate(colours):
                line = stacked_lines.stacked_line_for_indices(
                    row_index=i_row,
                    flux_index=i_flux,
                    background_index=i_background,
                    date_index=0)
                
                if line.n_stacked >= 3:

                    #
                    # Compile data into easy form to fit
                    y_all = np.append(y_all, np.array(line.model_trail)) 
                    noise_all_temp = np.append(noise_all_temp, np.array(line.model_trail_noise))
                    generated_trails.append(line.model_full_trail_untrailed)
                    n_lines_used += 1
    if n_lines_used == 0:
        return None, None, np.zeros(ut.trail_length)
    
    # Do the noiseall adjustment fix
    noise_all = np.sqrt(np.square(noise_all_temp)+const_fix**2)  

    # Calculated the weighted mean for the trail heights
    weighing = np.array([])
    for errors in noise_all:
        weighing = np.append(weighing, 1/(errors)**2)
    mean_height = np.sum((weighing*y_all))/np.sum(weighing)
    global mean_height_before
    mean_height_before=mean_height
    
    
    # Duplicate the x arrays for all trails
    x_all = np.tile(np.arange(ut.trail_length) + 1, n_lines_used)
    #x_one=np.array(np.arange(ut.trail_length)+1)

    
    # Make instance of analysis, passing it the data.  
    analysis = Analysis(
       x=x_all,
       y=y_all,
       noise=noise_all,
       generated_trails=generated_trails
    )
    
    #plt.plot(analysis.x, analysis.y, label='Analysis x and y')
    
    # Load our optimiser
    dynesty = af.DynestyStatic(number_of_cores=16, sample="rwalk", walks=10, nlive=500,
                               iterations_per_update=10000000)#, #force_x1_cpu=True)
    
    print(dynesty.config_dict_run)
    #exit(dynesty.config_dict_search)
    
    # Do the fitting
    print('Perfoming global AUTOFIT: ')
    result = dynesty.fit(
    model=model,
    analysis=analysis,
    )
    
    print(f"log likelihood = {result.log_likelihood}")
    
    best_trail_model = result.instance
    
    global result_info_pre
    result_info_pre=result.info
    print(result.info)

    print(f"beta = {best_trail_model.beta}")
    print(f"rho_q = {best_trail_model.rho_q}")
    print(f"a = {best_trail_model.a}")
    print(f"b = {best_trail_model.b}")
    print(f"c = {best_trail_model.c}")
    print(f"tau_a = {best_trail_model.tau_a}")
    print(f"tau_b = {best_trail_model.tau_b}")
    print(f"tau_c = {best_trail_model.tau_c}")
    print(f"notch = {best_trail_model.notch}")
    
    # Make sure the best fit results are global variables so we can call them later
    global best_fit_beta
    global best_fit_rho_q
    global best_fit_a
    global best_fit_b
    global best_fit_c
    global best_fit_tau_a
    global best_fit_tau_b
    global best_fit_tau_c
    global best_fit_notch
    global best_fit_loglikelihood
    
    best_fit_loglikelihood=result.log_likelihood
    best_fit_beta=best_trail_model.beta
    best_fit_rho_q=best_trail_model.rho_q
    best_fit_a=best_trail_model.a
    best_fit_b=best_trail_model.b
    best_fit_c=best_trail_model.c
    best_fit_tau_a=best_trail_model.tau_a
    best_fit_tau_b=best_trail_model.tau_b
    best_fit_tau_c=best_trail_model.tau_c
    best_fit_notch=best_trail_model.notch
    best_fit_mean_height=mean_height
    
    

    for i_row in range(n_row_bins):
        for i_flux in range(n_flux_bins):
            # Furthest row bin at the top
            ax = axes[n_row_bins - 1 - i_row][i_flux]

            # Plot each background bin's stack
            for i_background, c in enumerate(colours):
                line = stacked_lines.stacked_line_for_indices(
                    row_index=i_row,
                    flux_index=i_flux,
                    background_index=i_background,
                    date_index=0
                )
                # Skip empty and single-entry bins
                if line.n_stacked <= 1:
                    continue
                
                # Don't plot the warm pixel itself
                
                
                
                trail = np.array(line.model_trail)  # + line.model_background
                noise = np.sqrt(np.square(line.model_trail_noise)+const_fix**2)  # + line.model_background
                #noise_small = line.model_trail_noise
                
                #adjusted_flux=line.mean_flux + 12*N_plotted
                #adjusted_bg=line.mean_background - N_plotted

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
# =============================================================================
#                     ax.errorbar(
#                         pixels[where_pos],
#                         trail[where_pos],
#                         yerr=noise_small[where_pos],
#                         color='yellow',
#                         capsize=2,
#                         alpha=0.7,
#                     )
#                     ax.errorbar(
#                         pixels[where_neg],
#                         abs(trail[where_neg]),
#                         yerr=noise_small[where_neg],
#                         color='yellow',
#                         fmt=",",
#                         alpha=0.7,
#                         zorder=-2,
#                     )
# =============================================================================
                print('Plotting one autofit subplot...')
                global_autofit=trail_model_arctic_notch_pushed_plot(x=pixels, 
                                           rho_q=float(best_trail_model.rho_q), 
                                           generated_trails=line.model_full_trail_untrailed,
                                           beta=float(best_trail_model.beta), 
                                           w=w, 
                                           A=float(best_trail_model.a), 
                                           B=float(best_trail_model.b), 
                                           C=float(best_trail_model.c), 
                                           tau_a=float(best_trail_model.tau_a), 
                                           tau_b=float(best_trail_model.tau_b), 
                                           tau_c=float(best_trail_model.tau_c),
                                           notch=float(best_trail_model.notch)
                                          )
                print('Done!')

                
                ax.plot(pixels, global_autofit, color='red', ls='-.', alpha=0.7)
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
                        "%d" % stacked_lines.row_bins[i_row + 1],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="left",
                        va="center",
                    )
            if i_row == n_row_bins - 1:
                if i_flux == 0:
                    ax.text(
                        0.3,
                        1.01,
                        r"e$^-$ Flux:",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                    )
                flux_max = stacked_lines.flux_bins[i_flux + 1]
                pow10 = np.floor(np.log10(flux_max))
                text = r"$%.1f \!\times\! 10^{%d}$" % (flux_max / 10 ** pow10, pow10)
                ax.text(
                    1.0, 1.01, text, transform=ax.transAxes, ha="center", va="bottom"
                )
            if i_row == int(n_row_bins / 2) and i_flux == n_flux_bins - 1:
                text = "Background (e$^-$):  "
                for i_background in range(n_background_bins):
                    text += "%.0f$-$%.0f" % (
                        stacked_lines.background_bins[i_background],
                        stacked_lines.background_bins[i_background + 1],
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
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print("Saved", save_path.name)
        
    print("Total fit processing time: ", time.time() - start_time, "seconds")
    
    #  Print results to csv file 
    writefilename=f"{dataset_date}_notch_pushed_{const_fix}" 
    with open(writefilename+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"Log likelihood = {result.log_likelihood}"])
        writer.writerow([f"beta = {best_trail_model.beta}"])
        writer.writerow([f"rho_q = {best_trail_model.rho_q}"])
        writer.writerow([f"a = {best_trail_model.a}"])
        writer.writerow([f"b = {best_trail_model.b}"])
        writer.writerow([f"c = {best_trail_model.c}"])
        writer.writerow([f"tau_a = {best_trail_model.tau_a}"])
        writer.writerow([f"tau_b = {best_trail_model.tau_b}"])
        writer.writerow([f"tau_c = {best_trail_model.tau_c}"])
        writer.writerow([f"notch = {best_trail_model.notch}"])
        writer.writerow([f"mean height = {mean_height}"])
        writer.writerow([result.info])
       
            
    print("Data file written!")
    
    # Put the csv file into the appropriate folder
    batch_path = path.join(path.sep, "cosma", "home", "durham","dc-barr6", "warm_pixels_workspace", 
                           "batch_scripts")
    csv_directory = Path(batch_path)
    csvs_all=list(pathlib.Path(csv_directory).glob('*.csv'))
    csvs_string=[]
    for stuff in csvs_all:
        csvs_string.append(str(stuff))
    csv_list=[x for x in csvs_string if f"{dataset_date}_notch_pushed_{const_fix}" in x]
    print(csv_list)
    csv_name=str(os.path.basename(csv_list[0]))
    print(csv_name)
    target2=path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo", f"notch_pushed_{const_fix}", "csv_files",
                     str(csv_name))
    shutil.copyfile(csv_list[0],target2)
    
    
    
    # Now apply the correction to the images
    print("Commencing image correction")
    
    
# Import data to be fitted
start_time2=time.time()
cosma_path = path.join(path.sep, "cosma5", "data", "durham", "rjm")
#dataset_folder="Paolo's_03_2020"
#dataset_name="03_2020"

cosma_dataset_path = path.join(cosma_path, "paolo", "datasets", dataset_date)
cosma_output_path = path.join(cosma_path, "paolo",f"notch_pushed_{const_fix}")
workspace_path = "/cosma5/data/durham/rjm/paolo/dc-barr6/warm_pixels_workspace/"
#config_path = path.join(workspace_path, "cosma", "config")

dataset_directory=Path(cosma_dataset_path)


dataset = wp.Dataset(dataset_directory)

group = dataset.group("ABCD")

# Create the directory where we will save all the outputs
dir = os.path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo", f"notch_pushed_{const_fix}")
if not os.path.exists(dir):
    os.mkdir(dir)

dir = os.path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo", f"notch_pushed_{const_fix}",
                 f"{dataset_date}_notch_pushed_{const_fix}")
if not os.path.exists(dir):
    os.mkdir(dir)
    
dir = os.path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo", f"notch_pushed_{const_fix}",
                 "csv_files")
if not os.path.exists(dir):
    os.mkdir(dir)
    
data_directory = dataset_directory

# Find all the fits files.
temp_files_all=list(pathlib.Path(data_directory).glob('*.fits'))
temp_files_string=[]
for stuff in temp_files_all:
    temp_files_string.append(str(stuff))
    
temp_files_string_short=[x for x in temp_files_string if not 'bia' in x]
temp_files=[]
for stuff in temp_files_string_short:
    temp_files.append(Path(stuff))


for file in temp_files:
    print(file)
    temp_image_path=file
    
    # Load each quadrant of the image  (see pypi.org/project/autoarray)
    print('Loading image: ', temp_image_path)
    ImageACS.get_MJD(
            file_path=temp_image_path,
            quadrant_letter="D",
        )

# Call the 50 plot function we just defined    
Paolo_autofit_global_50(
    group,
    save_path=Path(path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo", f"notch_pushed_{const_fix}",
                     f"{dataset_date}_notch_pushed_{const_fix}"))/f"{dataset_date}_notch_pushed_{const_fix}.png"
)
 

# Now do the image correction using the best fit values from the fitting
import arcticpy as arctic
#import autoarray as aa
#from pathlib import Path
#import os 
#import pathlib
#from os import path
#import shutil


data_directory = dataset_directory

# Find all the fits files. Correct _bia images first. 
files_all=list(pathlib.Path(data_directory).glob('*.fits'))
files_string=[]
for stuff in files_all:
    files_string.append(str(stuff))
    
print('Now correcting _bia files')
files_string_bia=[x for x in files_string if 'bia' in x]
files_bia=[]
for stuff in files_string_bia:
    files_bia.append(Path(stuff))
    
for file in files_bia:
    print(file)
    image_path=file
    density_a=best_fit_a*best_fit_rho_q*rho_fix
    density_b=best_fit_b*best_fit_rho_q*rho_fix
    density_c=best_fit_c*best_fit_rho_q*rho_fix
    
    # Load each quadrant of the image  (see pypi.org/project/autoarray)
    print('Loading image: ', image_path)
    image_A, image_B, image_C, image_D = [
        paolo_ImageACS.from_fits(
            file_path=image_path,
            quadrant_letter=quadrant,
            bias_subtract_via_bias_file=False,
            bias_subtract_via_prescan=False,
        ).native
        for quadrant in ["A", "B", "C", "D"]
    ]
    
    traps = [
         arctic.TrapInstantCapture(density=density_a, release_timescale=best_fit_tau_a),
         arctic.TrapInstantCapture(density=density_b, release_timescale=best_fit_tau_b),
         arctic.TrapInstantCapture(density=density_c, release_timescale=best_fit_tau_c),      
     ]
    print('Passing fit parameters to arCTIc')
    roe = arctic.ROE()
    ccd = arctic.CCD(full_well_depth=84700, well_notch_depth=best_fit_notch, well_fill_power=best_fit_beta)
    
    
    # Remove CTI  (see remove_cti() in src/cti.cpp)
    print('Removing CTI')
    image_out_A, image_out_B, image_out_C, image_out_D = [
        arctic.remove_cti(
               image=image,
               n_iterations=5,
               parallel_roe=roe,
               parallel_ccd=ccd,
               parallel_traps=traps,
               parallel_express=5,
               verbosity=0,
        )
        for image in [image_A, image_B, image_C, image_D]
    ]
    
    filename=str(os.path.basename(file))
    output_path = path.join(path.sep, "cosma5", "data", "durham", "rjm","paolo", f"notch_pushed_{const_fix}", 
                            f"{dataset_date}_notch_pushed_{const_fix}", filename)
    
    # Save the corrected image
    print('Saving image',output_path)
    paolo_output_quadrants_to_fits(
        file_path=output_path,
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
    print('Corrected image saved!')

# Now correct the science images
files_string_short=[x for x in files_string if not 'bia' in x]
files=[]
for stuff in files_string_short:
    files.append(Path(stuff))


for file in files:
    print(file)
    image_path=file
    density_a=best_fit_a*best_fit_rho_q*rho_fix
    density_b=best_fit_b*best_fit_rho_q*rho_fix
    density_c=best_fit_c*best_fit_rho_q*rho_fix
    
    # Load each quadrant of the image  (see pypi.org/project/autoarray)
    print('Loading image: ', image_path)
    image_A, image_B, image_C, image_D = [
        ImageACS.from_fits(
            file_path=image_path,
            quadrant_letter=quadrant,
            bias_subtract_via_bias_file=True,
            bias_subtract_via_prescan=True,
            bias_file_path=path.join(path.sep, "cosma5", "data", "durham", "rjm","paolo", f"notch_pushed_{const_fix}", 
                                    f"{dataset_date}_notch_pushed_{const_fix}")
        ).native
        for quadrant in ["A", "B", "C", "D"]
    ]
    
    traps = [
         arctic.TrapInstantCapture(density=density_a, release_timescale=best_fit_tau_a),
         arctic.TrapInstantCapture(density=density_b, release_timescale=best_fit_tau_b),
         arctic.TrapInstantCapture(density=density_c, release_timescale=best_fit_tau_c),      
     ]
    print('Passing fit parameters to arCTIc')
    roe = arctic.ROE()
    ccd = arctic.CCD(full_well_depth=84700, well_notch_depth=best_fit_notch, well_fill_power=best_fit_beta)
    
    
    # Remove CTI  (see remove_cti() in src/cti.cpp)
    print('Removing CTI')
    image_out_A, image_out_B, image_out_C, image_out_D = [
        arctic.remove_cti(
               image=image,
               n_iterations=5,
               parallel_roe=roe,
               parallel_ccd=ccd,
               parallel_traps=traps,
               parallel_express=5,
               verbosity=0,
        )
        for image in [image_A, image_B, image_C, image_D]
    ]
    
    filename=str(os.path.basename(file))
    output_path = path.join(path.sep, "cosma5", "data", "durham", "rjm","paolo", f"notch_pushed_{const_fix}", 
                            f"{dataset_date}_notch_pushed_{const_fix}", filename)
    
    # Save the corrected image
    print('Saving image',output_path)
    output_quadrants_to_fits(
        file_path=output_path,
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
    print('Corrected image saved!')
    
    
print('All images corrected!')

print("Total correction processing time: ", time.time() - start_time2, "seconds")


# Now do the 50 plot diagram for the corrected dataset and it's best fit model.
def Paolo_autofit_global_50_after(group: QuadrantGroup, use_corrected=False, save_path=None): 
    
    stacked_lines = group.stacked_lines()
    
    #date = stacked_lines.date How can I get the date value from stacked_lines? 
    
    # Define constants and free variables
    # CCD
    beta = 0.556
    w = 84700.0
    # Trap species
    a = 0.180
    b = 0.789
    c = 0.032
    # Trap lifetimes before or after the temperature change
# =============================================================================
#     if date < ut.date_T_change:
#         tau_a = 0.48
#         tau_b = 4.86
#         tau_c = 20.6
#     else:
#         tau_a = 0.74
#         tau_b = 7.70
#         tau_c = 37.0
#     
# =============================================================================
# =============================================================================
#     tau_a = 0.74
#     tau_b = 7.70
#     tau_c = 37.0
#     sigma_a = 1.0
#     sigma_b = 1.0
#     sigma_c = 1.0
#     notch=0.0
# =============================================================================
    
    # CCD
    rho_q = af.UniformPrior(
        lower_limit=-10.0,
        #lower_limit=0.0,
        upper_limit=10.0,
    )
# =============================================================================
#     beta = af.GaussianPrior(
#               mean=0.478,
#               sigma=0.1,
#        )
# =============================================================================
    
    # Trap species
# =============================================================================
#     a = af.UniformPrior(
#         lower_limit=0,
#         upper_limit=1.0,
#     )
#     b = af.UniformPrior(
#         lower_limit=0,
#         upper_limit=1.0,
#     )
#     c = 1 - (a + b)
# =============================================================================
    
# =============================================================================
#     tau_a = af.GaussianPrior(
#           mean=tau_a,
#           sigma=0.2,
#       )
#     tau_b = af.GaussianPrior(
#           mean=tau_b,
#           sigma=2.0,
#       )
#     tau_c = af.GaussianPrior(
#           mean=tau_c,
#           sigma=10.0,
#       )
# =============================================================================
# =============================================================================
#     sigma_a = af.LogUniformPrior(lower_limit=0.0001, upper_limit=100.0)
#     sigma_b = af.LogUniformPrior(lower_limit=0.0001, upper_limit=100.0)
#     sigma_c = af.LogUniformPrior(lower_limit=0.0001, upper_limit=100.0)
# =============================================================================
# =============================================================================
#     notch = af.UniformPrior(
#         lower_limit=-10000.0,
#         #lower_limit=0.0,
#         upper_limit=10000.0,
#     )
# =============================================================================
    
    beta = best_fit_beta
    a = best_fit_a
    b = best_fit_b
    c = best_fit_c
    tau_a = best_fit_tau_a
    tau_b = best_fit_tau_b
    tau_c = best_fit_tau_c
    notch = best_fit_notch


    model = af.Model(
        TrailModel,
        days_var=days_var,
        rho_q=rho_q,
        beta=beta,
        w=w,
        a=a,
        b=b,
        c=c,
        tau_a=tau_a,
        tau_b=tau_b,
        tau_c=tau_c,
        notch=notch
    )
    
# =============================================================================
#     model.add_assertion(c > 0.0)
#     model.add_assertion(tau_a > 0.0)
#     model.add_assertion(tau_b > 0.0)
#     model.add_assertion(tau_c > 0.0)
# =============================================================================
    
    
    # Extract row bins
    n_row_bins = stacked_lines.n_row_bins
    n_flux_bins = stacked_lines.n_flux_bins
    n_background_bins = stacked_lines.n_background_bins

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
    fontsize = 20

    
    # Compile the data from all stacked lines for the global AUTOFIT
    n_lines_used = 0
    y_all = np.array([])
    #x_all = np.array([])
    noise_all_temp = np.array([])
    noise_all = np.array([])
    generated_trails=[]
   #x_one=np.array(np.arange(ut.trail_length)+1)

    for i_row in range(n_row_bins):
        for i_flux in range(n_flux_bins):
            ax = axes[n_row_bins - 1 - i_row][i_flux]
            for i_background, c in enumerate(colours):
                line = stacked_lines.stacked_line_for_indices(
                    row_index=i_row,
                    flux_index=i_flux,
                    background_index=i_background,
                    date_index=0)
                
                if line.n_stacked >= 3:

                    #
                    # Compile data into easy form to fit
                    #
                    
                   
                    y_all = np.append(y_all, np.array(line.model_trail))
                    
                    noise_all_temp = np.append(noise_all_temp, np.array(line.model_trail_noise))
                    generated_trails.append(line.model_full_trail_untrailed_abs)
                    n_lines_used += 1
    if n_lines_used == 0:
        return None, None, np.zeros(ut.trail_length)
    
    # Do the noiseall adjustment fix
    noise_all = np.sqrt(np.square(noise_all_temp)+const_fix**2)  
    
    # Calculated the weighted mean for the trail heights
    weighing = np.array([])
    for errors in noise_all:
        weighing = np.append(weighing, 1/(errors)**2)
    mean_height = np.sum((weighing*y_all))/np.sum(weighing)
    

    # Duplicate the x arrays for all trails
    x_all = np.tile(np.arange(ut.trail_length) + 1, n_lines_used)
    #x_one=np.array(np.arange(ut.trail_length)+1)

    
    
    # Make instance of analysis, passing it the data.  
    analysis = Analysis(
       x=x_all,
       y=y_all,
       noise=noise_all,
       generated_trails=generated_trails
    )
    
    #plt.plot(analysis.x, analysis.y, label='Analysis x and y')
    
    # Load our optimiser
    dynesty = af.DynestyStatic(number_of_cores=16, sample="rwalk", walks=10, nlive=500, 
                               iterations_per_update=10000000)#, #force_x1_cpu=True)
    
    print(dynesty.config_dict_run)
    #exit(dynesty.config_dict_search)
    
    # Do the fitting
    print('Perfoming global AUTOFIT: ')
    result = dynesty.fit(
    model=model,
    analysis=analysis,
    )
    
    print(f"log likelihood = {result.log_likelihood}")
    
    best_trail_model = result.instance
    print(result.info)

    print(f"beta = {best_trail_model.beta}")
    print(f"rho_q = {best_trail_model.rho_q}")
    print(f"a = {best_trail_model.a}")
    print(f"b = {best_trail_model.b}")
    print(f"c = {best_trail_model.c}")
    print(f"tau_a = {best_trail_model.tau_a}")
    print(f"tau_b = {best_trail_model.tau_b}")
    print(f"tau_c = {best_trail_model.tau_c}")
    print(f"notch = {best_trail_model.notch}")
 
    # Calculate the percent reduction in mean height and rho_q:
    mean_height_reduction = (mean_height_before-mean_height)/mean_height_before
    rho_q_reduction = (best_fit_rho_q-best_trail_model.rho_q)/best_fit_rho_q
    

    for i_row in range(n_row_bins):
        for i_flux in range(n_flux_bins):
            # Furthest row bin at the top
            ax = axes[n_row_bins - 1 - i_row][i_flux]

            # Plot each background bin's stack
            for i_background, c in enumerate(colours):
                line = stacked_lines.stacked_line_for_indices(
                    row_index=i_row,
                    flux_index=i_flux,
                    background_index=i_background,
                    date_index=0
                )
                # Skip empty and single-entry bins
                if line.n_stacked <= 1:
                    continue
                
                # Don't plot the warm pixel itself
                
             
                trail = line.model_trail  # + line.model_background
                noise = np.sqrt(np.square(line.model_trail_noise)+const_fix**2)  # + line.model_background
                #noise_small = line.model_trail_noise
                
                #adjusted_flux=line.mean_flux + 12*N_plotted
                #adjusted_bg=line.mean_background - N_plotted

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
# =============================================================================
#                 ax.errorbar(
#                     pixels[where_pos],
#                     trail[where_pos],
#                     yerr=noise_small[where_pos],
#                     color='yellow',
#                     capsize=2,
#                     alpha=0.7,
#                 )
#                 ax.errorbar(
#                     pixels[where_neg],
#                     abs(trail[where_neg]),
#                     yerr=noise_small[where_neg],
#                     color='yellow',
#                     fmt=",",
#                     alpha=0.7,
#                     zorder=-2,
#                 )
# =============================================================================


                print('Plotting one autofit subplot...')
                global_autofit=trail_model_arctic_notch_pushed_plot(x=pixels, 
                                           rho_q=float(best_trail_model.rho_q), 
                                           generated_trails=line.model_full_trail_untrailed_abs,
                                           beta=float(best_fit_beta), 
                                           w=w, 
                                           A=float(best_fit_a), 
                                           B=float(best_fit_b), 
                                           C=float(best_fit_c), 
                                           tau_a=float(best_fit_tau_a), 
                                           tau_b=float(best_fit_tau_b), 
                                           tau_c=float(best_fit_tau_c),
                                           notch=float(best_fit_notch)
                                          )
                print('Done!')

                
                ax.plot(pixels, global_autofit, color='red', ls='-.', alpha=0.7)
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
                        "%d" % stacked_lines.row_bins[i_row + 1],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="left",
                        va="center",
                    )
            if i_row == n_row_bins - 1:
                if i_flux == 0:
                    ax.text(
                        0.3,
                        1.01,
                        r"e$^-$ Flux:",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                    )
                flux_max = stacked_lines.flux_bins[i_flux + 1]
                pow10 = np.floor(np.log10(flux_max))
                text = r"$%.1f \!\times\! 10^{%d}$" % (flux_max / 10 ** pow10, pow10)
                ax.text(
                    1.0, 1.01, text, transform=ax.transAxes, ha="center", va="bottom"
                )
            if i_row == int(n_row_bins / 2) and i_flux == n_flux_bins - 1:
                text = "Background (e$^-$):  "
                for i_background in range(n_background_bins):
                    text += "%.0f$-$%.0f" % (
                        stacked_lines.background_bins[i_background],
                        stacked_lines.background_bins[i_background + 1],
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
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print("Saved", save_path.name)
        
    print("Total post correction fit processing time: ", time.time() - start_time, "seconds")
    
    #  Print results to csv file 
    writefilename=f"{dataset_date}_notch_pushed_{const_fix}_corrected"
    with open(writefilename+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"MJD = {MJD_var}"])
        writer.writerow([f"Log likelihood before = {best_fit_loglikelihood}"])
        writer.writerow([f"Log likelihood after = {result.log_likelihood}"])
        writer.writerow([f"beta = {best_trail_model.beta}"])
        writer.writerow([f"rho_q before = {best_fit_rho_q}"])
        writer.writerow([f"rho_q after = {best_trail_model.rho_q}"])
        writer.writerow([f"a = {best_trail_model.a}"])
        writer.writerow([f"b = {best_trail_model.b}"])
        writer.writerow([f"c = {best_trail_model.c}"])
        writer.writerow([f"tau_a = {best_trail_model.tau_a}"])
        writer.writerow([f"tau_b = {best_trail_model.tau_b}"])
        writer.writerow([f"tau_c = {best_trail_model.tau_c}"])
        writer.writerow([f"notch = {best_trail_model.notch}"])
        writer.writerow([f"mean height = {mean_height}"])
        writer.writerow([f"mean height reduction = {mean_height_reduction}"])
        writer.writerow([f"rho_q reduction = {rho_q_reduction}"])
        writer.writerow([f"CCDGAIN = {CCDGAIN_var}"])
        writer.writerow([result.info])
            
    print("Data file written!")
    
    # Put the csv file into the output folder
    batch_path = path.join(path.sep, "cosma", "home", "durham", "dc-barr6",
                           "warm_pixels_workspace", "batch_scripts")
    csv_directory = Path(batch_path)
    csvs_all=list(pathlib.Path(csv_directory).glob('*.csv'))
    csvs_string=[]
    for stuff in csvs_all:
        csvs_string.append(str(stuff))
    csv_list=[x for x in csvs_string if f"{dataset_date}_notch_pushed_{const_fix}_corrected" in x]
    print(csv_list)
    csv_name=str(os.path.basename(csv_list[0]))
    print(csv_name)
    target3=path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo",f"notch_pushed_{const_fix}",
                     "csv_files", str(csv_name))
    shutil.copyfile(csv_list[0],target3)

# Import data to be fitted
cosma_dataset_path = path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo",f"notch_pushed_{const_fix}",
                               f"{dataset_date}_notch_pushed_{const_fix}")
cosma_output_path = cosma_dataset_path
workspace_path = "/cosma5/data/durham/rjm/paolo/dc-barr6/warm_pixels_workspace/"
#config_path = path.join(workspace_path, "cosma", "config")

dataset_directory=Path(cosma_dataset_path)


dataset = wp.Dataset(dataset_directory)

group = dataset.group("ABCD")


# Call the 50 plot function we just defined    
Paolo_autofit_global_50_after(
    group,
    save_path=Path(path.join(path.sep, "cosma5", "data", "durham", "rjm", "paolo", f"notch_pushed_{const_fix}",
                     f"{dataset_date}_notch_pushed_{const_fix}"))/f"{dataset_date}_notch_pushed_{const_fix}_corrected.png"
)

