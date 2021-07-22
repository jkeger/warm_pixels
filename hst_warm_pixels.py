"""
Find, stack, and plot warm pixels in multiple sets of HST ACS images.

For each dataset of images:
+ For each image (and quadrant):
    + Find possible warm pixels
    + Find consistent warm pixels
+ Plot distributions of the warm pixels
+ Stack the warm pixel trails in bins
+ Plot the stacked trails

By default, runs all the functions for the chosen list of image datasets,
skipping any that have been run before and saved their output. Use the optional
flags to choose manually which functions to run.

Parameters
----------
dataset_list : str (opt.)
    The name of the list of image datasets to run. Defaults to "test". See the
    dataset_lists dictionary for the options.

--mdate_old_* : str (opt.)
    A "year/month/day" requirement to remake files saved/modified before this
    date. Defaults to only check whether a file already exists. Alternatively,
    set "1" to force remaking or "0" to force not.

    --mdate_old_fwp, -f
        Find warm pixels.

    --mdate_old_cwp, -c
        Consistent warm pixels.

    --mdate_old_dwp, -d
        Distributions of warm pixels in histograms.

    --mdate_old_swp, -s
        Stacked warm pixels.

    --mdate_old_pst, -p
        Plot stacked trails.

    --mdate_old_all, -a
        Sets the default for all others, can be overridden individually.

--quadrants, -q : str (opt.)
    The image quadrants to use, e.g. "A" or "ABCD" (default).

--test_image_and_bias_files, -t : str (opt.)
    Test loading the image and corresponding bias files in the list of datasets.
"""

import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lmfit
import argparse
import time
import datetime
from astropy.time import Time

from pixel_lines import PixelLine, PixelLineCollection
from warm_pixels import find_warm_pixels
from misc import *  # Plotting defaults etc

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path, "../PyAutoArray/"))
import autoarray as aa


# ========
# Constants
# ========
# Number of pixels for each trail, not including the warm pixel itself
trail_length = 8
# Julian dates
date_acs_launch = 2452334.5  # ACS launched, SM3B, 01 March 2002
date_T_change = 2453920.0  # Temperature changed, 03 July 2006
date_side2_fail = 2454128.0  # ACS stopped working, 27 January 2007
date_sm4_repair = 2454968.0  # ACS repaired, SM4, 16 May 2009
# Convert to days since ACS launch
day_T_change = date_T_change - date_acs_launch
day_side2_fail = date_side2_fail - date_acs_launch
day_sm4_repair = date_sm4_repair - date_acs_launch


# ========
# Image datasets
# ========
class Dataset(object):
    def __init__(self, name):
        """Simple class to store a list of image file paths and mild metadata.

        Parameters
        ----------
        name : str
            The name of the dataset, i.e. the name of the directory containing the
            image files, assumed to be in dataset_root.

        Attributes
        ----------
        path : str
            File path to the dataset directory.

        image_names : [str]
        image_paths : [str]
            The list of image file names, excluding and including the full path
            and extension, respectively.

        saved_stacked_lines, saved_stacked_info : str
            The file names for saving and loading derived data, including the
            path.
        """
        self.name = name
        self.path = dataset_root + self.name + "/"

        # Image file paths
        files = os.listdir(dataset_root + self.name)
        self.image_names = [f[:-5] for f in files if f[-9:] == "_raw.fits"]
        self.image_paths = [self.path + name + ".fits" for name in self.image_names]
        self.n_images = len(self.image_names)

        # Save paths
        self.saved_stacked_lines = self.path + "saved_stacked_lines.pickle"
        self.saved_stacked_info = self.path + "saved_stacked_info.npz"
        self.plotted_stacked_trails = (
            path + "/stacked_trail_plots/%s_stacked_trails.png" % self.name
        )
        self.plotted_distributions = self.path + "plotted_distributions.png"

    @property
    def date(self):
        """Return the Julian date of the set, taken from the first image."""
        image = aa.acs.ImageACS.from_fits(
            file_path=self.image_paths[0], quadrant_letter="A"
        )
        return 2400000.5 + image.header.modified_julian_date

    def saved_lines(self, quadrant):
        """Return the file name including the path for saving derived data."""
        return self.path + "saved_lines_%s.pickle" % quadrant

    def saved_consistent_lines(self, quadrant):
        """Return the file name including the path for saving derived data."""
        return self.path + "saved_consistent_lines_%s.pickle" % quadrant


dataset_root = os.path.join(path, "../hst_acs_datasets/")
datasets_pre_T_change = [
    # Date, days since acs launch
    # Aidan
    "01_2003",  # 2003/01/16, 321
    "05_2004",  # 2004/05/13, 804
    "10_2004",  # 2004/11/07, 982
    "12_2004",  # 2004/12/15, 1020
    "04_2005",  # 2005/04/05, 1131
    "05_2005",  # 2005/05/14, 1170
    "09_2005",  # 2005/09/04, 1283
    "11_2005",  # 2005/11/14, 1354
    "04_2006",  # 2006/04/28, 1519
    # Richard
    "shortSNe0",  # 2002/06/22, 113  ## Aidan: 1step error
    "shortSNe1",  # 2002/11/20, 264
    "shortSNe2",  # 2003/05/06, 431  ## Aidan: something wrong
    "shortSNe3",  # 2003/05/08, 433
    "shortSNe4",  # 2003/05/12, 437
    "early",  # 2003/10/20, 598
    "middle1",  # 2004/04/30, 791
    "middle2",  # 2004/05/20, 811
    "ten1a",  # 2004/05/24, 815
    "ten1b",  # 2004/05/29, 820
    "late",  # 2005/03/29, 1124
    "later",  # 2005/05/12, 1168
    "ten2a",  # 2005/09/22, 1301
    "ten2b",  # 2006/02/09, 1441
    "richmassey60680",  # 2006/02/13, 1445
    "richmassey60493",  # 2006/02/13, 1445
    "longSNe5",  # 2006/02/21, 1453
    "longSNe6",  # 2006/03/19, 1479
    "longSNe4",  # 2006/04/04, 1495
    "shortSNe5",  # 2006/04/04, 1495
    "shortSNe6",  # 2006/04/13, 1504
    "shortSNe7",  # 2006/04/23, 1514
    "longSNe3",  # 2006/05/15, 1536
    "shortSNe8",  # 2006/05/15, 1536
]
datasets_post_T_change = [
    # Date, days since acs launch
    # Aidan
    "07_2006",  # 2006/07/05, 1587
    "05_2010",  # 2010/05/25, 3007
    "04_2011",  # 2011/04/04, 3321
    "05_2012",  # 2012/05/27, 3740
    "04_2013",  # 2013/04/02, 4050
    "02_2014",  # 2014/02/23, 4377
    "04_2014",  # 2014/04/19, 4432
    "01_2015",  # 2015/01/21, 4709
    "06_2015",  # 2015/06/16, 4855
    "09_2015",  # 2015/09/01, 4932
    "01_2016",  # 2016/01/05, 5058
    "05_2016",  # 2016/05/23, 5197
    "09_2016",  # 2016/09/24, 5321
    "04_2017",  # 2017/04/05, 5514
    "06_2017",  # 2017/06/25, 5595
    "08_2017",  # 2017/08/08, 5639
    "10_2017",  # 2017/10/03, 5695
    "02_2018",  # 2018/02/16, 5831
    "08_2018",  # 2018/08/12, 6008
    "12_2018",  # 2018/12/05, 6123
    "01_2019",  # 2019/01/07, 6156
    "04_2019",  # 2019/04/01, 6240
    "07_2019",  # 2019/07/16, 6346
    "07_2019_2",  # 2019/07/15, 6345
    "10_2019",  # 2019/10/30, 6452
    "11_2019",  # 2019/11/19, 6472
    "11_2019_2",  # 2019/11/16, 6469
    "12_2019",  # 2019/12/06, 6489
    "12_2019_2",  # 2019/12/31, 6514
    "01_2020",  # 2020/01/04, 6518
    "03_2020",  # 2020/03/21, 6595
    "04_2020",  # 2020/04/12, 6617
    "07_2020",  # 2020/07/31, 6727
    "08_2020_1",  # 2020/08/07, 6734
    "09_2020_2",  # 2020/09/17, 6775
    "12_2020",  # 2020/12/03, 6852
    # Richard
    "longSNe2",  # 2006/07/13, 1595
    "richmassey60494",  # 2006/07/13, 1595
    "richmassey60490",  # 2006/07/17, 1599  ## e.g. low
    "richmassey61093",  # 2006/07/31, 1613  ## e.g. high
    "richmassey60491",  # 2006/08/16, 1629  ## e.g. low
    "ten3",  # 2006/08/18, 1631
    "shortSNe9",  # 2006/08/18, 1631
    "richmassey60488",  # 2006/08/22, 1635
    "richmassey60489",  # 2006/08/22, 1635
    "richmassey61092",  # 2006/09/11, 1655  ## e.g. high
    "longSNe1",  # 2006/09/16, 1660
    "shortSNeA",  # 2006/09/16, 1660
    "ten4",  # 2006/11/09, 1714
    "richmassey60487",  # 2006/12/07, 1742
    "richmassey60492",  # 2006/12/07, 1742
    "richmassey60484",  # 2006/12/11, 1746
    "richmassey60486",  # 2006/12/23, 1758
    "richmassey60485",  # 2006/12/31, 1766
    "sm41",  # 2009/08/26, 2735
    "sm42",  # 2009/08/26, 2735
    "sm43",  # 2009/11/02, 2803
    "sm44",  # 2009/11/08, 2809
    "sm45",  # 2010/01/23, 2885
    "richmassey72704",  # 2010/02/11, 2904
    "richmassey72703",  # 2010/02/17, 2910
    "richmassey72702",  # 2010/03/07, 2928
    "richmassey72701",  # 2010/04/09, 2961
    "richmassey72700",  # 2010/04/18, 2970
    "richmassey72699",  # 2010/04/22, 2974
    "richmassey72698",  # 2010/05/10, 2992
    "obama",  # 2010/07/08, 3051
    "huff_spt814a",  # 2011/10/04, 3504
    "huff_spt606a",  # 2011/10/06, 3506
    "huff_spt606f",  # 2011/11/27, 3558
    "huff_spt606g",  # 2011/12/03, 3564
    "huff_spt606b",  # 2012/01/04, 3596
    "huff_spt606c",  # 2012/01/20, 3612
    "huff_spt606d",  # 2012/03/03, 3655
    "huff_spt606e",  # 2012/07/17, 3791
    "huff_spt814b",  # 2012/07/25, 3799  ## Aidan: something wrong
    "huff_spt814c",  # 2012/10/14, 3880
    "huff_spt606h",  # 2012/10/22, 3888
    "candels2013b",  # 2013/01/02, 3960
    "candels2013a",  # 2013/01/02, 3960
    "obama2",  # 2013/04/17, 4065
]
datasets_all = np.append(datasets_pre_T_change, datasets_post_T_change)
datasets_test = ["12_2020"]
datasets_test_2 = [
    "richmassey60490",
    # "richmassey61093",
]
# Dictionary of list names
dataset_lists = {
    "test": datasets_test,
    "test_2": datasets_test_2,
    "pre_T_change": datasets_pre_T_change,
    "post_T_change": datasets_post_T_change,
    "all": datasets_all,
}
# Convert all to Dataset objects
for key in dataset_lists.keys():
    dataset_lists[key] = [Dataset(dataset) for dataset in dataset_lists[key]]


# ========
# Utility functions
# ========
def prep_parser():
    """Prepare the sys args parser."""
    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument(
        "dataset_list",
        nargs="?",
        default="test",
        type=str,
        help="The list of image datasets to run.",
    )

    # Date requirements for re-making files
    parser.add_argument(
        "-a",
        "--mdate_old_all",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for all saved files.",
    )
    parser.add_argument(
        "-f",
        "--mdate_old_fwp",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for found warm pixels.",
    )
    parser.add_argument(
        "-c",
        "--mdate_old_cwp",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for consistent warm pixels.",
    )
    parser.add_argument(
        "-d",
        "--mdate_old_dwp",
        default=None,
        type=str,
        required=False,
        help="Distributions of warm pixels in histograms.",
    )
    parser.add_argument(
        "-s",
        "--mdate_old_swp",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for stacked warm pixels.",
    )
    parser.add_argument(
        "-p",
        "--mdate_old_pst",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for plot stacked trails.",
    )

    # Other options
    parser.add_argument(
        "-q",
        "--quadrants",
        default="ABCD",
        type=str,
        required=False,
        help="The image quadrants to use.",
    )
    parser.add_argument(
        "-t",
        "--test_image_and_bias_files",
        action="store_true",
        default=False,
        required=False,
        help="Test loading the image and corresponding bias files.",
    )

    return parser


def need_to_make_file(filepath, date_old=None):
    """Return True if a file needs to be (re)made.

    Parameters
    ----------
    filepath : str
        The file that might need to be remade.

    date_old : str (opt.)
        A "year/month/day"-format date requirement to remake files saved before
        this date. Defaults to only check whether a file already exists.
        Alternatively, set "1" to force remaking or "0" to force not.
    """
    # If the file doesn't exist
    if not os.path.isfile(filepath):
        return True

    # If the file was saved too long ago
    if date_old is not None:
        # Overrides
        if date_old == "1":
            return True
        elif date_old == "0":
            return False

        # Compare with modified date
        time_mod = os.path.getmtime(filepath)
        time_old = time.mktime(
            datetime.datetime.strptime(date_old, "%Y/%m/%d").timetuple()
        )
        if time_mod < time_old:
            return True

    return False


def dec_yr_to_jd(dates):
    """Convert one or more decimal-year dates to Julian dates."""
    time = Time(dates, format="decimalyear")
    time.format = "jd"
    return time.value


def jd_to_dec_yr(dates):
    """Convert one or more Julian dates to decimal-year dates."""
    time = Time(dates, format="jd")
    time.format = "decimalyear"
    return time.value


def test_image_and_bias_files(dataset):
    """Test loading the set of image and corresponding bias files.

    Missing bias files can be downloaded from ssb.stsci.edu/cdbs_open/cdbs/jref.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    Returns
    -------
    all_okay : bool
        True if no errors, False if any errors hit.
    """
    all_okay = True

    for i_image in range(dataset.n_images):
        image_path = dataset.image_paths[i_image]
        image_name = dataset.image_names[i_image]
        print("\r  %s " % image_name, end="", flush=True)

        # Load the image
        try:
            image = aa.acs.ImageACS.from_fits(
                file_path=image_path,
                quadrant_letter="A",
                bias_subtract_via_bias_file=True,
                bias_subtract_via_prescan=True,
            ).native
        except FileNotFoundError as e:
            all_okay = False
            print(str(e))

    return all_okay


# ========
# Main functions
# ========
def find_dataset_warm_pixels(dataset, quadrant="A"):
    """Find the possible warm pixels in all images in a dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrant : str (opt.)
        The quadrant (A, B, C, D) of the image to load.

    Saves
    -----
    warm_pixels : PixelLineCollection
        The set of warm pixel trails, saved to dataset.saved_lines(quadrant).
    """
    # Initialise the collection of warm pixel trails
    warm_pixels = PixelLineCollection()
    print("")

    # Find the warm pixels in each image
    for i_image in range(dataset.n_images):
        image_path = dataset.image_paths[i_image]
        image_name = dataset.image_names[i_image]
        print(
            "    %s_%s (%d of %d): "
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

        date = 2400000.5 + image.header.modified_julian_date

        image_name_q = image_name + "_%s" % quadrant

        # Find the warm pixel trails
        new_warm_pixels = find_warm_pixels(
            image=image, trail_length=trail_length, origin=image_name_q, date=date
        )
        print("Found %d possible warm pixels " % len(new_warm_pixels))

        # Plot
        plot_warm_pixels(
            image,
            PixelLineCollection(new_warm_pixels),
            save_path=dataset.path + image_name_q,
        )

        # Add them to the collection
        warm_pixels.append(new_warm_pixels)

    # Save
    warm_pixels.save(dataset.saved_lines(quadrant))


def find_consistent_warm_pixels(dataset, quadrant="A"):
    """Find the consistent warm pixels in a dataset.

    find_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrant : str (opt.)
        The quadrant (A, B, C, D) of the image to load.

    Saves
    -----
    warm_pixels : PixelLineCollection
        The set of consistent warm pixel trails, saved to
        dataset.saved_consistent_lines(quadrant).
    """
    # Load
    warm_pixels = PixelLineCollection()
    warm_pixels.load(dataset.saved_lines(quadrant))

    # Find the warm pixels present in at least 2/3 of the images
    consistent_lines = warm_pixels.find_consistent_lines(fraction_present=2 / 3)
    print(
        "Found %d consistents of %d possibles"
        % (len(consistent_lines), warm_pixels.n_lines)
    )

    # Extract the consistent warm pixels
    warm_pixels.lines = warm_pixels.lines[consistent_lines]

    # Save
    warm_pixels.save(dataset.saved_consistent_lines(quadrant))


def stack_dataset_warm_pixels(dataset, quadrants=["A"]):
    """Stack a set of premade warm pixel trails into bins.

    find_dataset_warm_pixels() and find_consistent_warm_pixels() must first be
    run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    quadrants : [str]
        The list of quadrants (A, B, C, D) of the images to load.

    Saves
    -----
    stacked_lines : PixelLineCollection
        The set of stacked pixel trails, saved to dataset.saved_stacked_lines.
    """
    # Load
    warm_pixels = PixelLineCollection()
    # Append data from each quadrant
    for quadrant in quadrants:
        warm_pixels.load(dataset.saved_consistent_lines(quadrant))

    # Subtract preceeding pixels in each line before stacking
    for i in range(warm_pixels.n_lines):
        warm_pixels.lines[i].data[trail_length + 1 :] -= warm_pixels.lines[i].data[
            :trail_length
        ][::-1]

    # Stack the lines in bins by distance from readout and total flux
    n_row_bins = 5
    n_flux_bins = 10
    n_background_bins = 1
    (
        stacked_lines,
        row_bins,
        flux_bins,
        date_bins,
        background_bins,
    ) = warm_pixels.generate_stacked_lines_from_bins(
        n_row_bins=n_row_bins,
        n_flux_bins=n_flux_bins,
        n_background_bins=n_background_bins,
        return_bin_info=True,
    )
    print("Stacked lines in %d bins" % (n_row_bins * n_flux_bins * n_background_bins))

    # Save
    stacked_lines.save(dataset.saved_stacked_lines)
    np.savez(
        dataset.saved_stacked_info, row_bins, flux_bins, date_bins, background_bins
    )


def trail_model(x, rho_q, n_e, n_bg, row, alpha, w, A, B, C, tau_a, tau_b, tau_c):
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

    alpha : float
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
    return (
        rho_q
        * ((n_e / w) ** alpha - (n_bg / w) ** alpha)
        * row
        * (
            A * np.exp((1 - x) / tau_a) * (1 - np.exp(-1 / tau_a))
            + B * np.exp((1 - x) / tau_b) * (1 - np.exp(-1 / tau_b))
            + C * np.exp((1 - x) / tau_c) * (1 - np.exp(-1 / tau_c))
        )
    )


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
    alpha = 0.478
    w = 84700.0
    # Trap species
    A = 0.17
    B = 0.45
    C = 0.38
    # Trap lifetimes before or after the temperature change
    if date < date_T_change:
        tau_a = 0.48
        tau_b = 4.86
        tau_c = 20.6
    else:
        tau_a = 0.74
        tau_b = 7.70
        tau_c = 37.0

    return trail_model(x, rho_q, n_e, n_bg, row, alpha, w, A, B, C, tau_a, tau_b, tau_c)


def fit_total_trap_density(x_all, y_all, noise_all, n_e_all, n_bg_all, row_all, date):
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
        The Background number of electrons (e-).

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
    model = lmfit.models.Model(
        func=trail_model_hst,
        independent_vars=["x", "n_e", "n_bg", "row", "date"],
        nan_policy="propagate",  ## not "omit"? If any needed at all?
    )
    params = model.make_params()

    # Initialise the fit
    params["rho_q"].value = 0.1
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
    # print(result.fit_report())

    return result.params.get("rho_q").value, result.params.get("rho_q").stderr


def fit_dataset_total_trap_density(dataset):
    """Load, prep, and pass the stacked-trail data to fit_total_trap_density().

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    Returns
    -------
    rho_q : float
        The best-fit total number density of traps per pixel.

    rho_q_std : float
        The standard error on the total trap density.
    """
    # Load
    stacked_lines = PixelLineCollection()
    stacked_lines.load(dataset.saved_stacked_lines)
    npzfile = np.load(dataset.saved_stacked_info)
    row_bins, flux_bins, date_bins, background_bins = [
        npzfile[var] for var in npzfile.files
    ]
    n_row_bins = len(row_bins) - 1
    n_flux_bins = len(flux_bins) - 1
    n_date_bins = len(date_bins) - 1
    n_background_bins = len(background_bins) - 1

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
    # Skip the lowest-row and lowest-flux bins
    for i_row in range(1, n_row_bins):
        for i_flux in range(1, n_flux_bins):
            for i_background in range(n_background_bins):
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
                    y_all = np.append(y_all, line.data[-trail_length:])
                    noise_all = np.append(noise_all, line.noise[-trail_length:])
                    n_e_each = np.append(n_e_each, line.mean_flux)
                    n_bg_each = np.append(n_bg_each, line.mean_background)
                    row_each = np.append(row_each, line.mean_row)
                    n_lines_used += 1

    # Duplicate the x arrays for all trails
    x_all = np.tile(np.arange(trail_length) + 1, n_lines_used)

    # Duplicate the single parameters of each trail for all pixels
    n_e_all = np.repeat(n_e_each, trail_length)
    n_bg_all = np.repeat(n_bg_each, trail_length)
    row_all = np.repeat(row_each, trail_length)

    # Run the fitting
    rho_q, rho_q_std = fit_total_trap_density(
        x_all, y_all, noise_all, n_e_all, n_bg_all, row_all, dataset.date
    )

    return rho_q, rho_q_std


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

    im = plt.imshow(X=image, aspect="equal", vmin=0, vmax=500)
    plt.scatter(
        warm_pixels.locations[:, 1],
        warm_pixels.locations[:, 0],
        marker=".",
        c="r",
        edgecolor="none",
        s=0.1,
        alpha=0.7,
    )

    plt.colorbar(im)
    plt.axis("off")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=800)
        plt.close()


def plot_warm_pixel_distributions(dataset, quadrants=["A"], save_path=None):
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
        colours = A1_c[: len(quadrants)]
    else:
        colours = ["k"]

    # Load
    warm_pixels = PixelLineCollection()
    # Append data from each quadrant
    for quadrant in quadrants:
        warm_pixels.load(dataset.saved_consistent_lines(quadrant))

    # Set bins for all quadrants
    n_row_bins = 20
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

    date_min = np.amin(warm_pixels.dates - date_acs_launch)
    date_max = np.amax(warm_pixels.dates - date_acs_launch)
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
            warm_pixels.dates - date_acs_launch, bins=date_bins
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

    nice_plot(ax1)
    nice_plot(ax2)
    nice_plot(ax3)
    nice_plot(ax4)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
        print("Saved", save_path[-40:])


def plot_stacked_trails(dataset, save_path=None):
    """Plot a tiled set of stacked trails.

    stack_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    save_path : str (opt.)
        The file path for saving the figure. If None, then show the figure.
    """
    # Load
    stacked_lines = PixelLineCollection()
    stacked_lines.load(dataset.saved_stacked_lines)
    npzfile = np.load(dataset.saved_stacked_info)
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
    pixels = np.arange(1, trail_length + 1)
    y_min = np.amin(abs(stacked_lines.data[:, -trail_length:]))
    y_max = 4 * np.amax(stacked_lines.data[:, -trail_length:])
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
    rho_q_set, rho_q_std_set = fit_dataset_total_trap_density(dataset)

    # Plot each stack
    for i_row in range(n_row_bins):
        for i_flux in range(n_flux_bins):
            # Furthest row bin at the top
            ax = axes[n_row_bins - 1 - i_row][i_flux]

            for i_background, c in enumerate(colours):
                bin_index = PixelLineCollection.stacked_bin_index(
                    i_row=i_row,
                    n_row_bins=n_row_bins,
                    i_flux=i_flux,
                    n_flux_bins=n_flux_bins,
                    i_background=i_background,
                    n_background_bins=n_background_bins,
                )

                line = stacked_lines.lines[bin_index]

                # Skip empty and single-entry bins
                if line.n_stacked <= 1:
                    continue

                # Don't plot the warm pixel itself
                trail = line.data[-trail_length:]
                noise = line.noise[-trail_length:]

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
                    facecolor="none",
                    marker="o",
                    alpha=0.7,
                )

                # ========
                # Plot fitted trail
                # ========
                # Fit the total trap density to this single stacked trail
                rho_q, rho_q_std = fit_total_trap_density(
                    x_all=pixels,
                    y_all=trail,
                    noise_all=noise,
                    n_e_all=line.mean_flux,
                    n_bg_all=line.mean_background,
                    row_all=line.mean_row,
                    date=dataset.date,
                )
                model_pixels = np.linspace(1, trail_length, 20)
                model_trail = trail_model_hst(
                    x=model_pixels,
                    rho_q=rho_q,
                    n_e=line.mean_flux,
                    n_bg=line.mean_background,
                    row=line.mean_row,
                    date=dataset.date,
                )
                ax.plot(model_pixels, model_trail, color=c, ls="--", alpha=0.7)
                # Also plot the full-set fit
                model_trail = trail_model_hst(
                    x=model_pixels,
                    rho_q=rho_q_set,
                    n_e=line.mean_flux,
                    n_bg=line.mean_background,
                    row=line.mean_row,
                    date=dataset.date,
                )
                ax.plot(model_pixels, model_trail, color=c, ls=":", alpha=0.7)

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

            ax.set_xlim(0.5, trail_length + 0.5)
            ax.set_xticks(np.arange(2, trail_length + 0.1, 2))
            ax.set_xticks(np.arange(1, trail_length + 0.1, 2), minor=True)
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
                ax.set_ylabel("Charge (e$^-$)")

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
                        r"Flux (e$^-$):",
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
            if n_background_bins > 1:
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
                ax.text(
                    0.03,
                    0.03,
                    r"$\rho_{\rm q} = %.2g \pm %.2g$" % (rho_q_set, rho_q_std_set),
                    transform=ax.transAxes,
                    size=fontsize,
                    ha="left",
                    va="bottom",
                )

            # Tidy
            if i_row == 0 and i_flux == 0:
                set_large_ticks(ax)
            elif i_row == 0:
                set_large_ticks(ax, do_x=False)
            elif i_flux == 0:
                set_large_ticks(ax, do_y=False)
            set_font_size(ax)

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print("Saved", save_path[-40:])


# ========
# Main
# ========
if __name__ == "__main__":
    # ========
    # Parse arguments
    # ========
    parser = prep_parser()
    args = parser.parse_args()

    list_name = args.dataset_list
    if list_name not in dataset_lists.keys():
        print("Error: Invalid dataset_list", list_name)
        print("  Choose from:", list(dataset_lists.keys()))
        raise ValueError
    dataset_list = dataset_lists[list_name]

    if args.mdate_old_all is not None:
        if args.mdate_old_fwp is None:
            args.mdate_old_fwp = args.mdate_old_all
        if args.mdate_old_cwp is None:
            args.mdate_old_cwp = args.mdate_old_all
        if args.mdate_old_dwp is None:
            args.mdate_old_dwp = args.mdate_old_all
        if args.mdate_old_swp is None:
            args.mdate_old_swp = args.mdate_old_all
        if args.mdate_old_pst is None:
            args.mdate_old_pst = args.mdate_old_all

    quadrants = [q for q in args.quadrants]

    # Test loading the image and corresponding bias files
    if args.test_image_and_bias_files:
        print("Testing image and bias files...")
        all_okay = True

        for dataset in dataset_list:
            if not test_image_and_bias_files(dataset):
                all_okay = False
        print("")

        if not all_okay:
            exit()

    # ========
    # Find and stack warm pixels in each dataset
    # ========
    for i_dataset, dataset in enumerate(dataset_list):
        print(
            'Dataset "%s" (%d of %d in "%s", %d images, quadrant(s) %s)'
            % (
                dataset.name,
                i_dataset + 1,
                len(dataset_list),
                list_name,
                dataset.n_images,
                args.quadrants,
            )
        )

        # Find warm pixels in each image quadrant
        for quadrant in quadrants:
            # Find possible warm pixels in each image
            if need_to_make_file(
                dataset.saved_lines(quadrant), date_old=args.mdate_old_fwp
            ):
                print(
                    "  Find possible warm pixels (%s)..." % quadrant,
                    end=" ",
                    flush=True,
                )
                find_dataset_warm_pixels(dataset, quadrant)

            # Consistent warm pixels in the set
            if need_to_make_file(
                dataset.saved_consistent_lines(quadrant), date_old=args.mdate_old_cwp
            ):
                print(
                    "  Consistent warm pixels (%s)..." % quadrant, end=" ", flush=True
                )
                find_consistent_warm_pixels(dataset, quadrant)

        # Distributions of warm pixels in the set
        if need_to_make_file(
            dataset.plotted_distributions, date_old=args.mdate_old_dwp
        ):
            print("  Distributions of warm pixels...", end=" ", flush=True)
            plot_warm_pixel_distributions(
                dataset, quadrants, save_path=dataset.plotted_distributions
            )

        # Stack in bins
        if need_to_make_file(dataset.saved_stacked_lines, date_old=args.mdate_old_swp):
            print("  Stack warm pixel trails...", end=" ", flush=True)
            stack_dataset_warm_pixels(dataset, quadrants)

        # Plot stacked lines
        if need_to_make_file(
            dataset.plotted_stacked_trails, date_old=args.mdate_old_pst
        ):
            print("  Plot stacked trails...", end=" ", flush=True)
            plot_stacked_trails(dataset, save_path=dataset.plotted_stacked_trails)
