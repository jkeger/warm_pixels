"""
Find, stack, and plot warm pixels in multiple sets of HST ACS images.

Parameters
----------
dataset_list : str (opt.)
    The name of the list of image datasets to run. Defaults to "test". See the
    dataset_list_names dictionary for the options.

--date_old_* : str (opt.)
    A "year/month/day"-format date requirement to remake files saved before this
    date. Defaults to only check whether a file already exists. Alternatively,
    set "1" to force remaking or "0" to force not.
    --date_old_all, -a
        Overrides all others.
    --date_old_fwp, -f
        Find warm pixels.
    --date_old_cwp, -c
        Consistent warm pixels.
    --date_old_swp, -s
        Stacked warm pixels.
    --date_old_ttd, -t
        Total trap density.
    --date_old_pst, -p
        Plot stacked trails.
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lmfit
import argparse
import time
import datetime

from pixel_lines import PixelLine, PixelLineCollection
from warm_pixels import find_warm_pixels

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path, "../PyAutoArray/"))
import autoarray as aa


# ========
# Constants
# ========
# Number of pixels for each trail, not including the warm pixel itself
trail_length = 8


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
            The list of image file names, excluding and including the full path and
            extension, respectively.

        bias_name : str
        bias_path : str
            The bias image file name, excluding and including the full path and
            extension, respectively.

        saved_lines, saved_consistent_lines, saved_stacked_lines, saved_stacked_info
            : str
            The file names for saving and loading derived data, including the path.
        """
        self.name = name
        self.path = dataset_root + self.name + "/"

        # Image file paths
        files = os.listdir(dataset_root + self.name)
        self.image_names = [f[:-5] for f in files if f[-9:] == "_raw.fits"]
        self.image_paths = [self.path + name + ".fits" for name in self.image_names]
        self.n_images = len(self.image_names)

        # Bias file path
        try:
            self.bias_name = [f[:-5] for f in files if f[-9:] == "_bia.fits"][0]
            self.bias_path = self.path + self.bias_name + ".fits"
        except IndexError:
            self.bias_name = None
            self.bias_path = None

        # Save paths
        self.saved_lines = self.path + "saved_lines.pickle"
        self.saved_consistent_lines = self.path + "saved_consistent_lines.pickle"
        self.saved_stacked_lines = self.path + "saved_stacked_lines.pickle"
        self.saved_stacked_info = self.path + "saved_stacked_info.npz"
        self.plotted_stacked_trails = self.path + "stacked_trails.png"

    @property
    def date(self):
        """Return the Julian date of the set, taken from the first image."""
        image = aa.acs.ImageACS.from_fits(
            file_path=self.image_paths[0], quadrant_letter="A"
        )
        return 2400000.5 + image.header.modified_julian_date


dataset_root = os.path.join(path, "../hst_acs_datasets/")
datasets_pre_2006_06 = [
    # In date order
    # Aidan
    "01_2003",
    "05_2004",
    "10_2004",
    "12_2004",
    "04_2005",
    "05_2005",
    "09_2005",
    "11_2005",
    "04_2006",
    # Richard
    "shortSNe0",  # Aidan: 1step error
    "shortSNe1",
    "shortSNe2",  # Aidan: something wrong
    "shortSNe3",
    "shortSNe4",
    "early",
    "middle1",
    "middle2",
    "ten1a",
    "ten1b",
    "late",
    "later",
    "ten2a",
    "ten2b",
    "richmassey60680",
    "richmassey60493",
    "longSNe5",
    "longSNe6",
    "longSNe4",
    "shortSNe5",
    "shortSNe6",
    "shortSNe7",
    "longSNe3",
    "shortSNe8",
]
datasets_post_2006_06 = [
    # Aidan
    "07_2006",
    # "09_2006", # Missing?
    "05_2010",
    "04_2011",
    "05_2012",
    "04_2013",
    "02_2014",
    "04_2014",
    "01_2015",
    "06_2015",
    "09_2015",
    "01_2016",
    "05_2016",
    "09_2016",
    "04_2017",
    "06_2017",
    "08_2017",
    "10_2017",
    "02_2018",
    "08_2018",
    "12_2018",
    "01_2019",
    "04_2019",
    "07_2019",
    "07_2019_2",
    "10_2019",
    "11_2019",
    "11_2019_2",
    "12_2019",
    "12_2019_2",
    "01_2020",
    "03_2020",
    "04_2020",
    "07_2020",
    "08_2020_1",
    "09_2020_2",
    "12_2020",
    # Richard
    "longSNe2",
    "richmassey60494",
    "richmassey60490",
    "richmassey61093",
    "richmassey60491",
    "ten3",
    "shortSNe9",
    "richmassey60488",
    "richmassey60489",
    "richmassey61092",
    "longSNe1",
    "shortSNeA",
    "ten4",
    "richmassey60487",
    "richmassey60492",
    "richmassey60484",
    "richmassey60486",
    "richmassey60485",
    "sm41",
    "sm42",
    "sm43",
    "sm44",
    "sm45",
    "richmassey72704",
    "richmassey72703",
    "richmassey72702",
    "richmassey72701",
    "richmassey72700",
    "richmassey72699",
    "richmassey72698",
    "obama",
    "huff_spt814a",
    "huff_spt606a",
    "huff_spt606f",
    "huff_spt606g",
    "huff_spt606b",
    "huff_spt606c",
    "huff_spt606d",
    "huff_spt606e",
    "huff_spt814b",  # Aidan: something wrong
    "huff_spt814c",
    "huff_spt606h",
    "candels2013b",
    "candels2013a",
    "obama2",
]
datasets_all = np.append(datasets_pre_2006_06, datasets_post_2006_06)
datasets_test = ["12_2020"]
datasets_test_2 = ["04_2011", "05_2012", "04_2013", "04_2014"]
# Dictionary of list names
dataset_list_names = {
    "test": datasets_test,
    "test_2": datasets_test_2,
    "pre_2006_06": datasets_pre_2006_06,
    "post_2006_06": datasets_post_2006_06,
    "all": datasets_all,
}
# Convert to Dataset objects
for key in dataset_list_names.keys():
    dataset_list_names[key] = [Dataset(dataset) for dataset in dataset_list_names[key]]


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
        "--date_old_all",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for all saved files.",
    )
    parser.add_argument(
        "-f",
        "--date_old_fwp",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for found warm pixels.",
    )
    parser.add_argument(
        "-c",
        "--date_old_cwp",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for consistent warm pixels.",
    )
    parser.add_argument(
        "-s",
        "--date_old_swp",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for stacked warm pixels.",
    )
    parser.add_argument(
        "-t",
        "--date_old_ttd",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for total trap density.",
    )
    parser.add_argument(
        "-p",
        "--date_old_pst",
        default=None,
        type=str,
        required=False,
        help="Oldest valid date for plot stacked trails.",
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


# ========
# Main functions
# ========
def find_dataset_warm_pixels(dataset):
    """Find the possible warm pixels in all images in a dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    Saves
    -----
    warm_pixels : PixelLineCollection
        The set of warm pixel trails, saved to dataset.saved_lines.
    """
    # Initialise the collection of warm pixel trails
    warm_pixels = PixelLineCollection()
    print("")

    # Find the warm pixels in each image
    for i_image in range(dataset.n_images):
        image_path = dataset.image_paths[i_image]
        image_name = dataset.image_names[i_image]
        print(
            "    %s (%d of %d): " % (image_name, i_image + 1, dataset.n_images),
            end="",
            flush=True,
        )

        # Load the image
        image = aa.acs.ImageACS.from_fits(
            file_path=image_path,
            quadrant_letter="A",
            bias_path=dataset.bias_path,
            bias_subtract_via_prescan=True,
        ).native

        date = 2400000.5 + image.header.modified_julian_date

        # Find the warm pixel trails
        new_warm_pixels = find_warm_pixels(
            image=image, trail_length=trail_length, origin=image_name, date=date
        )
        print("Found %d possible warm pixels" % len(new_warm_pixels))

        # Plot
        plot_warm_pixels(
            image,
            PixelLineCollection(new_warm_pixels),
            save_path=dataset.path + image_name,
        )

        # Add them to the collection
        warm_pixels.append(new_warm_pixels)

    # Save
    warm_pixels.save(dataset.saved_lines)


def find_consistent_warm_pixels(dataset):
    """Find the consistent warm pixels in a dataset.

    find_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    Saves
    -----
    warm_pixels : PixelLineCollection
        The set of consistent warm pixel trails, saved to
        dataset.saved_consistent_lines.
    """
    # Load
    warm_pixels = PixelLineCollection()
    warm_pixels.load(dataset.saved_lines)

    # Find the warm pixels present in at least 2/3 of the images
    consistent_lines = warm_pixels.find_consistent_lines(fraction_present=2 / 3)
    print(
        "Found %d consistent warm pixels out of %d possibles"
        % (len(consistent_lines), warm_pixels.n_lines)
    )

    # Extract the consistent warm pixels
    warm_pixels.lines = warm_pixels.lines[consistent_lines]

    # Save
    warm_pixels.save(dataset.saved_consistent_lines)
    print("")


def stack_dataset_warm_pixels(dataset):
    """Stack a set of premade warm pixel trails into bins.

    find_dataset_warm_pixels() and find_consistent_warm_pixels() must first be
    run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    Saves
    -----
    stacked_lines : PixelLineCollection
        The set of stacked pixel trails, saved to dataset.saved_stacked_lines.
    """
    # Load
    warm_pixels = PixelLineCollection()
    warm_pixels.load(dataset.saved_consistent_lines)

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


def trail_model_hst(x, rho_q, n_e, n_bg, row, is_pre_2006_06=False):
    """Wrapper for trail_model() for HST ACS.

    Parameters (where different to trail_model())
    ----------
    is_before_2006_06 : bool (opt.)
        Whether to use the pre-June 2006 or the post-June 2006 trap model.

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
    if is_pre_2006_06:
        tau_a = 0.48
        tau_b = 4.86
        tau_c = 20.6
    else:
        tau_a = 0.74
        tau_b = 7.70
        tau_c = 37.0

    return trail_model(x, rho_q, n_e, n_bg, row, alpha, w, A, B, C, tau_a, tau_b, tau_c)


def fit_total_trap_density(x_all, y_all, noise_all, n_e_all, n_bg_all, row_all):
    """WIP

    Takes the concatenated data from all trails to pass to trail_model() for
    fitting, so other than x and y, the parameters will be in subsets of the
    same value for the N consecutive pixels in each appended trail.

    Parameters
    ----------
    x_all : [float]
        The pixel position away from the trailed pixel, for each pixel in all
        trails.

    y_all : [float]
        The charge value, for each pixel in all trails.

    noise_all : [float]
        The charge noise error value, for each pixel in all trails.

    n_e_all : [float]
        The number of electrons in the trailed pixel's charge cloud (e-), for
        each pixel in all trails.

    n_bg_all : [float]
        The Background number of electrons (e-), for each pixel in all trails.

    row_all : [float]
        Distance in pixels of the trailed pixel from the readout register, for
        each pixel in all trails.

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
        independent_vars=["x", "n_e", "n_bg", "row"],
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
    )
    # print(result.fit_report())

    return result.params.get("rho_q").value, result.params.get("rho_q").stderr


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
        The file path for saving the figure. If None, the show the figure.
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


def plot_stacked_trails(dataset, save_path=None):
    """Plot a tiled set of stacked trails.

    stack_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    save_path : str (opt.)
        The file path for saving the figure. If None, the show the figure.
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

    # Date
    ##todo
    is_pre_2006_06 = False

    # Plot the stacked trails
    plt.figure(figsize=(25, 12))
    plt.subplots_adjust(wspace=0, hspace=0)
    gs = GridSpec(n_row_bins, n_flux_bins)
    axes = [
        [plt.subplot(gs[i_row, i_flux]) for i_flux in range(n_flux_bins)]
        for i_row in range(n_row_bins)
    ]

    # Don't plot the warm pixel itself
    pixels = np.arange(1, trail_length + 1)
    y_min = 0.8 * np.amin(abs(stacked_lines.data[:, -trail_length:]))
    y_max = 4 * np.amax(stacked_lines.data[:, -trail_length:])
    colours = plt.cm.jet(np.linspace(0.05, 0.95, n_background_bins))

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
                    n_e_all=np.ones(trail_length) * line.mean_flux,
                    n_bg_all=np.ones(trail_length) * line.mean_background,
                    row_all=np.ones(trail_length) * line.mean_row,
                )
                model_pixels = np.linspace(1, trail_length, 20)
                model_trail = trail_model_hst(
                    x=model_pixels,
                    rho_q=rho_q,
                    n_e=line.mean_flux,
                    n_bg=line.mean_background,
                    row=line.mean_row,
                    is_pre_2006_06=is_pre_2006_06,
                )
                ax.plot(model_pixels, model_trail, color=c, ls="--", alpha=0.7)

                # Annotate
                if i_background == 0:
                    text = "$N=%d$" % line.n_stacked
                else:
                    text = "\n" * i_background + "$%d$" % line.n_stacked
                ax.text(0.97, 0.96, text, transform=ax.transAxes, ha="right", va="top")

            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0.5, trail_length + 0.5)

            # Axis labels
            if i_flux == 0:
                ax.set_ylabel("Charge")
            else:
                ax.set_yticklabels([])
            if i_row == 0:
                ax.set_xlabel("Pixel")
                ax.set_xticks(np.arange(1, trail_length + 0.1, 2))
            else:
                ax.set_xticklabels([])

            # Bin labels
            if i_row == n_row_bins - 1:
                ax.xaxis.set_label_position("top")
                ax.set_xlabel(
                    "Flux:  %.2g$-$%.2g" % (flux_bins[i_flux], flux_bins[i_flux + 1])
                )
            if i_flux == n_flux_bins - 1:
                ax.yaxis.set_label_position("right")
                text = "Row:  %d$-$%d" % (row_bins[i_row], row_bins[i_row + 1])
                if i_row == int(n_row_bins / 2):
                    text += "\n\nBackground:  "
                    for i_background in range(n_background_bins):
                        text += "%.0f$-$%.0f" % (
                            background_bins[i_background],
                            background_bins[i_background + 1],
                        )
                        if i_background < n_background_bins - 1:
                            text += ",  "
                ax.set_ylabel(text)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200)
        print("Saved", save_path[-30:])


# ========
# Main
# ========
if __name__ == "__main__":
    # ========
    # Parse arguments
    # ========
    parser = prep_parser()
    args = parser.parse_args()

    if args.dataset_list not in dataset_list_names.keys():
        print("Error: Invalid dataset_list", args.dataset_list)
        print("  Choose from:", list(dataset_list_names.keys()))
        raise ValueError
    dataset_list = dataset_list_names[args.dataset_list]

    if args.date_old_all is not None:
        args.date_old_fwp = args.date_old_all
        args.date_old_cwp = args.date_old_all
        args.date_old_swp = args.date_old_all
        args.date_old_ttd = args.date_old_all
        args.date_old_pst = args.date_old_all

    # ========
    # Find and stack warm pixels in each dataset
    # ========
    for i_dataset, dataset in enumerate(dataset_list):
        print(
            'Dataset "%s" (%d of %d in "%s", %d images)'
            % (
                dataset.name,
                i_dataset + 1,
                len(dataset_list),
                args.dataset_list,
                dataset.n_images,
            )
        )

        # Warm pixels in each image
        if need_to_make_file(dataset.saved_lines, date_old=args.date_old_fwp):
            print("  Find possible warm pixels...", end=" ", flush=True)
            find_dataset_warm_pixels(dataset)

        # Consistent warm pixels in the set
        if need_to_make_file(
            dataset.saved_consistent_lines, date_old=args.date_old_cwp
        ):
            print("  Consistent warm pixels...", end=" ", flush=True)
            find_consistent_warm_pixels(dataset)

        # Stack in bins
        if need_to_make_file(dataset.saved_stacked_lines, date_old=args.date_old_swp):
            print("  Stack warm pixel trails...", end=" ", flush=True)
            stack_dataset_warm_pixels(dataset)

        # Plot stacked lines
        if need_to_make_file(
            dataset.plotted_stacked_trails, date_old=args.date_old_pst
        ):
            print("  Plot stacked trails...", end=" ", flush=True)
            plot_stacked_trails(dataset, save_path=dataset.plotted_stacked_trails)
