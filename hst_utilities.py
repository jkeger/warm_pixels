"""Utility functions etc. for hst_warm_pixels.py"""

import numpy as np
import os
import argparse
import time
import datetime
from astropy.time import Time

path = os.path.dirname(os.path.realpath(__file__))

dataset_root = os.path.join(path, "../hst_acs_datasets/")


# ========
# Constants
# ========
# Number of pixels for each trail, not including the warm pixel itself
trail_length = 8
# trail_length = 12
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

    # Optional arguments
    parser.add_argument(
        "-q",
        "--quadrants",
        default="ABCD",
        type=str,
        required=False,
        help="The image quadrants to use.",
    )

    # Date requirements for re-making files
    parser.add_argument(
        "-f",
        "--mdate_find",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for found warm pixels.",
    )
    parser.add_argument(
        "-c",
        "--mdate_consistent",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for consistent warm pixels.",
    )
    parser.add_argument(
        "-C",
        "--mdate_plot_consistent",
        default=None,
        type=str,
        required=False,
        help="Plot distributions of consistent warm pixels.",
    )
    parser.add_argument(
        "-s",
        "--mdate_stack",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for stacked warm pixels.",
    )
    parser.add_argument(
        "-S",
        "--mdate_plot_stack",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for plot stacked trails.",
    )
    parser.add_argument(
        "-a",
        "--mdate_all",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for all saved files.",
    )

    # Other functions
    parser.add_argument(
        "-r",
        "--prep_density",
        action="store_true",
        default=False,
        required=False,
        help="Fit the total trap density (rho_q) for all datasets.",
    )
    parser.add_argument(
        "-R",
        "--plot_density",
        action="store_true",
        default=False,
        required=False,
        help="Plot the evolution of the total trap density.",
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


def dataset_list_saved_density_evol(list_name, quadrants):
    """Return the file path for the saved density data for a dataset list."""
    return dataset_root + "density_evol_%s_%s.npz" % (list_name, "".join(quadrants))


def dataset_list_plotted_density_evol(list_name, quadrant_sets):
    """Return the file path for the saved density plot for a dataset list."""
    quadrant_label = ""
    for qs in quadrant_sets:
        quadrant_label += "_%s" % "".join(qs)
    return path + "/density_evol_%s%s.png" % (list_name, quadrant_label)
