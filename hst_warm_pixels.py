"""
WIP
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pixel_lines import PixelLine, PixelLineCollection
from warm_pixels import find_warm_pixels

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path, "../PyAutoArray/"))
import autoarray as aa


# ========
# Constants
# ========
trail_length = 9


# ========
# Image datasets
# ========
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
    "09_2006",
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


# ========
# Functions etc
# ========
class Dataset(object):
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

    def __init__(self, name):
        self.name = name
        self.path = dataset_root + self.name + "/"

        # Image file paths
        files = os.listdir(dataset_root + self.name)
        self.image_names = [f[:-5] for f in files if f[-9:] == "_raw.fits"]
        self.image_paths = [self.path + name + ".fits" for name in self.image_names]
        self.n_images = len(self.image_names)

        # Bias file path
        self.bias_name = [f[:-5] for f in files if f[-9:] == "_bia.fits"][0]
        self.bias_path = self.path + self.bias_name + ".fits"

        # Save paths
        self.saved_lines = self.path + "saved_lines"
        self.saved_consistent_lines = self.path + "saved_consistent_lines"
        self.saved_stacked_lines = self.path + "saved_stacked_lines"
        self.saved_stacked_info = self.path + "saved_stacked_info.npz"


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

    # Find the warm pixels in each image
    for i_image in range(dataset.n_images):
        image_path = dataset.image_paths[i_image]
        image_name = dataset.image_names[i_image]
        print(
            "  %s (%d of %d): " % (image_name, i_image + 1, dataset.n_images),
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


def plot_warm_pixels(image, warm_pixels, save_path=None):
    """Plot an image and mark the locations of warm pixels.

    stack_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    image : [[float]]
        The 2D image array.

    warm_pixels : PixelLineCollection
        The set of warm pixel trails.

    save_path : str
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
        # print("Saved", save_path[-40:])
        plt.close()


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
        warm_pixels.lines[i].data[trail_length:] -= warm_pixels.lines[i].data[
            : trail_length - 1
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


def plot_stacked_trails(dataset, save_path=None):
    """Plot a tiled set of stacked trails.

    stack_dataset_warm_pixels() must first be run for the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset object with a list of image file paths and metadata.

    save_path : str
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

    # Plot the stacked trails
    plt.figure(figsize=(25, 12))
    plt.subplots_adjust(wspace=0, hspace=0)
    gs = GridSpec(n_row_bins, n_flux_bins)
    axes = [
        [plt.subplot(gs[i_row, i_flux]) for i_flux in range(n_flux_bins)]
        for i_row in range(n_row_bins)
    ]

    # Don't plot the warm pixel itself
    pixels = np.arange(1, trail_length)
    y_min = np.amin(abs(stacked_lines.data[:, -trail_length + 1 :]))
    y_max = 2 * np.amax(stacked_lines.data[:, -trail_length + 1 :])
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
                trail = line.data[-trail_length + 1 :]
                noise = line.noise[-trail_length + 1 :]

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
                    marker="x",
                    alpha=0.7,
                )

                # ========
                # Plot fitted trail
                # ========
                ##

                # Annotate
                if i_background == 0:
                    text = "$N=%d$" % line.n_stacked
                else:
                    text = "\n" * i_background + "$%d$" % line.n_stacked
                ax.text(0.97, 0.96, text, transform=ax.transAxes, ha="right", va="top")

            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0.5, trail_length - 0.5)

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
        plt.savefig(save_path)
        print("Saved", save_path[-40:])
        plt.close()


# ========
# Main
# ========
if __name__ == "__main__":
    # ========
    # Find and stack warm pixels in each dataset
    # ========
    for name in datasets_test:
        dataset = Dataset(name)

        print('\nDataset "%s" (%d images)' % (name, dataset.n_images))

        # Warm pixels in each image
        if True:
            print("1. Find possible warm pixels")
            find_dataset_warm_pixels(dataset)

        # Consistent warm pixels in the set
        if True:
            print("2. Find consistent warm pixels")
            find_consistent_warm_pixels(dataset)

        # Stack in bins
        if True:
            print("3. Stack warm pixel trails")
            stack_dataset_warm_pixels(dataset)

        # Plot stacked lines
        if True:
            plot_stacked_trails(dataset, save_path=dataset.path + "stacked_trails")
