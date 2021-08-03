"""
Find, stack, and plot warm pixels in multiple datasets of HST ACS images.

See functions.py and utilities.py.

Full pipeline:
+ For each dataset of images:
    + For each image (and quadrant), find possible warm pixels
    + Find consistent warm pixels
    + Plot distributions of the warm pixels
    + Stack the warm pixel trails in bins
    + Plot the stacked trails
+ Fit the total trap density across all datasets
+ Plot the evolution of the total trap density

See hst_utilities.py to set parameters like the trail length and bin edges.

By default, runs all the functions for the chosen list of image datasets,
skipping any that have been run before and saved their output. Use the optional
flags to choose manually which functions to run.

Parameters
----------
dataset_list : str (opt.)
    The name of the list of image datasets to run. Defaults to "test". See the
    dataset_lists dictionary for the options.

--quadrants, -q : str (opt.)
    The image quadrants to use, e.g. "A" or "ABCD" (default). To analyse the
    quadrants separately (done regardless for functions before the stacking),
    use e.g. "A_B_C_D", or e.g. "AB_CD" for combined A & B kept separate from
    combined C & D.

--mdate_old_*, -* DATE : str (opt.)
    A "year/month/day" requirement to remake files saved/modified before this
    date. Defaults to only check whether a file already exists. Alternatively,
    set "1" to force remaking or "0" to force not.

    --mdate_find, -f
        Find warm pixels.

    --mdate_consistent, -c
        Consistent warm pixels.

    --mdate_plot_consistent, -C
        Plot distributions of consistent warm pixels.

    --mdate_stack, -s
        Stacked warm pixels.

    --mdate_plot_stack, -S
        Plot stacked trails.

    --mdate_all, -a
        Sets the default for all others, can be overridden individually.

--prep_density, -d
    Fit the total trap density across all datasets.

--plot_density, -D
    Plot the evolution of the total trap density.

--downsample, -w N i : int int
    Downsample the dataset list to run 1/N of the datasets, starting with set i.
    e.g. -w 10 5 will run the datasets with indices 5, 15, 25, ... in the list.

--test_image_and_bias_files, -t
    Test loading the image and corresponding bias files in the list of datasets.
"""

import numpy as np
import os
import sys

import hst_utilities as ut
import hst_functions as fu

sys.path.append(os.path.join(ut.path, "../PyAutoArray/"))
import autoarray as aa


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
        """
        self.name = name
        self.path = ut.dataset_root + self.name + "/"

        # Image file paths
        files = os.listdir(ut.dataset_root + self.name)
        self.image_names = [f[:-5] for f in files if f[-9:] == "_raw.fits"]
        self.image_paths = [self.path + name + ".fits" for name in self.image_names]
        self.n_images = len(self.image_names)

    @property
    def date(self):
        """Return the Julian date of the set, taken from the first image."""
        image = aa.acs.ImageACS.from_fits(
            file_path=self.image_paths[0], quadrant_letter="A"
        )
        return 2400000.5 + image.header.modified_julian_date

    # ========
    # File paths for saved data, including the quadrant(s)
    # ========
    def saved_lines(self, quadrant):
        """Return the file name including the path for saving derived data."""
        return self.path + "saved_lines_%s.pickle" % quadrant

    def saved_consistent_lines(self, quadrant):
        """Return the file name including the path for saving derived data."""
        return self.path + "saved_consistent_lines_%s.pickle" % quadrant

    def saved_stacked_lines(self, quadrants):
        """Return the file name including the path for saving derived data."""
        return self.path + "saved_stacked_lines_%s.pickle" % "".join(quadrants)

    def saved_stacked_info(self, quadrants):
        """Return the file name including the path for saving derived data."""
        return self.path + "saved_stacked_info_%s.npz" % "".join(quadrants)

    def plotted_stacked_trails(self, quadrants):
        """Return the file name including the path for saving derived data."""
        return ut.path + "/stacked_trail_plots/%s_plotted_stacked_trails_%s.png" % (
            self.name,
            "".join(quadrants),
        )

    def plotted_distributions(self, quadrants):
        """Return the file name including the path for saving derived data."""
        return ut.path + "/plotted_distributions/%s_plotted_distributions_%s.png" % (
            self.name,
            "".join(quadrants),
        )


datasets_pre_T_change = [
    # Aidan
    "01_2003",  # 2003/01/16, day 321, 7 images
    "05_2004",  # 2004/05/13, day 804, 12 images
    "10_2004",  # 2004/11/07, day 982, 8 images
    "12_2004",  # 2004/12/15, day 1020, 15 images
    "04_2005",  # 2005/04/05, day 1131, 8 images
    "05_2005",  # 2005/05/14, day 1170, 14 images
    "09_2005",  # 2005/09/04, day 1283, 12 images
    "11_2005",  # 2005/11/14, day 1354, 13 images  ## fit error?
    "04_2006",  # 2006/04/28, day 1519, 11 images
    # Richard
    "shortSNe1",  # 2002/11/20, day 264, 8 images
    "shortSNe2",  # 2003/05/06, day 431, 3 images
    "shortSNe3",  # 2003/05/08, day 433, 3 images
    "shortSNe4",  # 2003/05/12, day 437, 3 images
    "early",  # 2003/10/20, day 598, 23 images
    "middle1",  # 2004/04/30, day 791, 23 images
    "middle2",  # 2004/05/20, day 811, 21 images
    "ten1a",  # 2004/05/24, day 815, 15 images
    "ten1b",  # 2004/05/29, day 820, 10 images
    "late",  # 2005/03/29, day 1124, 19 images
    "later",  # 2005/05/12, day 1168, 21 images
    "ten2a",  # 2005/09/22, day 1301, 8 images
    "ten2b",  # 2006/02/09, day 1441, 8 images
    "richmassey60680",  # 2006/02/13, day 1445, 11 images
    "richmassey60493",  # 2006/02/13, day 1445, 11 images
    "longSNe5",  # 2006/02/21, day 1453, 28 images
    "longSNe6",  # 2006/03/19, day 1479, 20 images
    "longSNe4",  # 2006/04/04, day 1495, 28 images
    "shortSNe5",  # 2006/04/04, day 1495, 4 images
    "shortSNe6",  # 2006/04/13, day 1504, 3 images
    "shortSNe7",  # 2006/04/23, day 1514, 7 images
    "longSNe3",  # 2006/05/15, day 1536, 20 images
    "shortSNe8",  # 2006/05/15, day 1536, 3 images
]
datasets_post_T_change = [
    # Aidan
    "07_2006",  # 2006/07/05, day 1587, 7 images
    "05_2010",  # 2010/05/25, day 3007, 7 images
    "04_2011",  # 2011/04/04, day 3321, 15 images
    "05_2012",  # 2012/05/27, day 3740, 15 images
    "04_2013",  # 2013/04/02, day 4050, 15 images
    "02_2014",  # 2014/02/23, day 4377, 8 images
    "04_2014",  # 2014/04/19, day 4432, 10 images
    "01_2015",  # 2015/01/21, day 4709, 10 images
    "06_2015",  # 2015/06/16, day 4855, 24 images
    "09_2015",  # 2015/09/01, day 4932, 11 images
    "01_2016",  # 2016/01/05, day 5058, 8 images
    "05_2016",  # 2016/05/23, day 5197, 12 images
    "09_2016",  # 2016/09/24, day 5321, 8 images
    "04_2017",  # 2017/04/05, day 5514, 7 images
    "06_2017",  # 2017/06/25, day 5595, 10 images
    "08_2017",  # 2017/08/08, day 5639, 14 images
    "10_2017",  # 2017/10/03, day 5695, 5 images
    "02_2018",  # 2018/02/16, day 5831, 15 images
    "08_2018",  # 2018/08/12, day 6008, 14 images
    "12_2018",  # 2018/12/05, day 6123, 10 images
    "01_2019",  # 2019/01/07, day 6156, 12 images
    "04_2019",  # 2019/04/01, day 6240, 8 images
    "07_2019",  # 2019/07/16, day 6346, 10 images
    "07_2019_2",  # 2019/07/15, day 6345, 8 images
    "10_2019",  # 2019/10/30, day 6452, 8 images
    "11_2019_2",  # 2019/11/16, day 6469, 8 images
    "11_2019",  # 2019/11/19, day 6472, 15 images
    "12_2019",  # 2019/12/06, day 6489, 8 images
    "12_2019_2",  # 2019/12/31, day 6514, 10 images
    "01_2020",  # 2020/01/04, day 6518, 8 images
    "03_2020",  # 2020/03/21, day 6595, 6 images
    "04_2020",  # 2020/04/12, day 6617, 6 images
    "07_2020",  # 2020/07/31, day 6727, 8 images
    "08_2020_1",  # 2020/08/07, day 6734, 20 images
    "09_2020_2",  # 2020/09/17, day 6775, 14 images
    "12_2020",  # 2020/12/03, day 6852, 12 images
    # Richard
    "longSNe2",  # 2006/07/13, day 1595, 32 images
    "richmassey60494",  # 2006/07/13, day 1595, 9 images
    "richmassey60490",  # 2006/07/17, day 1599, 6 images
    "richmassey61093",  # 2006/07/31, day 1613, 8 images
    "richmassey60491",  # 2006/08/16, day 1629, 7 images
    "ten3",  # 2006/08/18, day 1631, 28 images
    "shortSNe9",  # 2006/08/18, day 1631, 5 images
    "richmassey60488",  # 2006/08/22, day 1635, 6 images
    "richmassey60489",  # 2006/08/22, day 1635, 5 images
    "richmassey61092",  # 2006/09/11, day 1655, 4 images
    "longSNe1",  # 2006/09/16, day 1660, 16 images
    "shortSNeA",  # 2006/09/16, day 1660, 3 images
    "ten4",  # 2006/11/09, day 1714, 18 images
    "richmassey60487",  # 2006/12/07, day 1742, 6 images
    "richmassey60492",  # 2006/12/07, day 1742, 6 images
    "richmassey60484",  # 2006/12/11, day 1746, 8 images
    "richmassey60486",  # 2006/12/23, day 1758, 9 images
    "richmassey60485",  # 2006/12/31, day 1766, 3 images
    "sm41",  # 2009/08/26, day 2735, 23 images
    "sm42",  # 2009/08/26, day 2735, 23 images
    "sm43",  # 2009/11/02, day 2803, 12 images
    "sm44",  # 2009/11/08, day 2809, 20 images
    "sm45",  # 2010/01/23, day 2885, 6 images
    "richmassey72704",  # 2010/02/11, day 2904, 16 images
    "richmassey72703",  # 2010/02/17, day 2910, 6 images
    "richmassey72702",  # 2010/03/07, day 2928, 8 images
    "richmassey72701",  # 2010/04/09, day 2961, 8 images
    "richmassey72700",  # 2010/04/18, day 2970, 24 images
    "richmassey72699",  # 2010/04/22, day 2974, 30 images
    "richmassey72698",  # 2010/05/10, day 2992, 4 images
    "obama",  # 2010/07/08, day 3051, 8 images
    "huff_spt814a",  # 2011/10/04, day 3504, 12 images
    "huff_spt606a",  # 2011/10/06, day 3506, 16 images
    "huff_spt606f",  # 2011/11/27, day 3558, 16 images
    "huff_spt606g",  # 2011/12/03, day 3564, 16 images
    "huff_spt606b",  # 2012/01/04, day 3596, 8 images
    "huff_spt606c",  # 2012/01/20, day 3612, 8 images
    "huff_spt606d",  # 2012/03/03, day 3655, 8 images
    "huff_spt606e",  # 2012/07/17, day 3791, 8 images
    "huff_spt814b",  # 2012/07/25, day 3799, 4 images
    "huff_spt814c",  # 2012/10/14, day 3880, 8 images
    "huff_spt606h",  # 2012/10/22, day 3888, 8 images
    "candels2013b",  # 2013/01/02, day 3960, 15 images
    "candels2013a",  # 2013/01/02, day 3960, 15 images
    "obama2",  # 2013/04/17, day 4065, 22 images
]
datasets_all = np.append(datasets_pre_T_change, datasets_post_T_change)
datasets_sample = [
    "shortSNe2",  # 2003/05/06, day 431, 3 images
    "05_2004",  # 2004/05/13, day 804, 12 images
    "04_2005",  # 2005/04/05, day 1131, 8 images
    "ten2b",  # 2006/02/09, day 1441, 8 images
    "04_2006",  # 2006/04/28, day 1519, 11 images
    # Temperature changed
    "richmassey60490",  # 2006/07/17, day 1599, 6 images
    "richmassey61093",  # 2006/07/31, day 1613, 8 images
    "richmassey60491",  # 2006/08/16, day 1629, 7 images
    "richmassey61092",  # 2006/09/11, day 1655, 4 images
    # Failure and repair
    "sm43",  # 2009/11/02, day 2803, 12 images
    "05_2010",  # 2010/05/25, day 3007, 7 images
    "04_2011",  # 2011/04/04, day 3321, 15 images
    "huff_spt814b",  # 2012/07/25, day 3799, 4 images
    "04_2013",  # 2013/04/02, day 4050, 15 images
    "02_2014",  # 2014/02/23, day 4377, 8 images
    "01_2015",  # 2015/01/21, day 4709, 10 images
    "01_2016",  # 2016/01/05, day 5058, 8 images
    "04_2017",  # 2017/04/05, day 5514, 7 images
    "10_2017",  # 2017/10/03, day 5695, 5 images
    "08_2018",  # 2018/08/12, day 6008, 14 images
    "04_2019",  # 2019/04/01, day 6240, 8 images
    "03_2020",  # 2020/03/21, day 6595, 6 images
    "12_2020",  # 2020/12/03, day 6852, 12 images
]
datasets_test = ["huff_spt814b"]
datasets_test_2 = [
    "richmassey60491",  # 2006/08/16, day 1629, 7 images
    "richmassey61092",  # 2006/09/11, day 1655, 4 images
    "04_2017",  # 2017/04/05, day 5514, 7 images
    "10_2017",  # 2017/10/03, day 5695, 5 images
]
# Dictionary of choosable list names
dataset_lists = {
    "all": datasets_all,
    "sample": datasets_sample,
    "test": datasets_test,
    "test_2": datasets_test_2,
}
# Convert all to Dataset objects
for key in dataset_lists.keys():
    dataset_lists[key] = [Dataset(dataset) for dataset in dataset_lists[key]]


# ========
# Main
# ========
if __name__ == "__main__":
    # ========
    # Parse arguments
    # ========
    parser = ut.prep_parser()
    args = parser.parse_args()

    # Datasets
    list_name = args.dataset_list
    if list_name not in dataset_lists.keys():
        print("Error: Invalid dataset_list", list_name)
        print("  Choose from:", list(dataset_lists.keys()))
        raise ValueError
    dataset_list = dataset_lists[list_name]

    # Split quadrants into separate or combined subsets
    # e.g. "AB_CD" --> [["A", "B"], ["C", "D"]]
    quadrant_sets = [[q for q in qs] for qs in args.quadrants.split("_")]
    # All quadrants, ignoring subsets
    all_quadrants = [q for qs in quadrant_sets for q in qs]

    # Date/override requirements
    if args.mdate_all is not None:
        if args.mdate_find is None:
            args.mdate_find = args.mdate_all
        if args.mdate_consistent is None:
            args.mdate_consistent = args.mdate_all
        if args.mdate_plot_consistent is None:
            args.mdate_plot_consistent = args.mdate_all
        if args.mdate_stack is None:
            args.mdate_stack = args.mdate_all
        if args.mdate_plot_stack is None:
            args.mdate_plot_stack = args.mdate_all

    # Downsample the dataset list
    if args.downsample is not None:
        N = int(args.downsample[0])
        i = int(args.downsample[1])
        dataset_list = dataset_list[i::N]
        downsample_print = "[%d::%d]" % (i, N)
    else:
        downsample_print = ""

    # Test loading the image and corresponding bias files
    if args.test_image_and_bias_files:
        print("Testing image and bias files...")
        all_okay = True

        for dataset in dataset_list:
            if not ut.test_image_and_bias_files(dataset):
                all_okay = False
        print("")

        if not all_okay:
            exit()

    # ========
    # Find and stack warm pixels in each dataset
    # ========
    for i_dataset, dataset in enumerate(dataset_list):
        print(
            'Dataset "%s" (%d of %d in "%s"%s, %d images, "%s")'
            % (
                dataset.name,
                i_dataset + 1,
                len(dataset_list),
                list_name,
                downsample_print,
                dataset.n_images,
                args.quadrants,
            )
        )

        # Find warm pixels in each image quadrant
        for quadrant in all_quadrants:
            # Find possible warm pixels in each image
            if ut.need_to_make_file(
                dataset.saved_lines(quadrant), date_old=args.mdate_find
            ):
                print(
                    "  Find possible warm pixels (%s)..." % quadrant,
                    end=" ",
                    flush=True,
                )
                fu.find_dataset_warm_pixels(dataset, quadrant)

            # Consistent warm pixels in the set
            if ut.need_to_make_file(
                dataset.saved_consistent_lines(quadrant), date_old=args.mdate_consistent
            ):
                print(
                    "  Consistent warm pixels (%s)..." % quadrant, end=" ", flush=True
                )
                fu.find_consistent_warm_pixels(
                    dataset,
                    quadrant,
                    flux_min=ut.flux_bins[0],
                    flux_max=ut.flux_bins[-1],
                )

        # Plot distributions of warm pixels in the set
        if ut.need_to_make_file(
            dataset.plotted_distributions(all_quadrants),
            date_old=args.mdate_plot_consistent,
        ):
            print("  Distributions of warm pixels...", end=" ", flush=True)
            fu.plot_warm_pixel_distributions(
                dataset,
                all_quadrants,
                save_path=dataset.plotted_distributions(all_quadrants),
            )

        # Stack warm pixels in each image quadrant or combined quadrants
        for quadrants in quadrant_sets:
            # Stack in bins
            if ut.need_to_make_file(
                dataset.saved_stacked_lines(quadrants), date_old=args.mdate_stack
            ):
                print(
                    "  Stack warm pixel trails (%s)..." % "".join(quadrants),
                    end=" ",
                    flush=True,
                )
                fu.stack_dataset_warm_pixels(dataset, quadrants)

            # Plot stacked lines
            if ut.need_to_make_file(
                dataset.plotted_stacked_trails(quadrants),
                date_old=args.mdate_plot_stack,
            ):
                print(
                    "  Plot stacked trails (%s)..." % "".join(quadrants),
                    end=" ",
                    flush=True,
                )
                fu.plot_stacked_trails(
                    dataset,
                    quadrants,
                    save_path=dataset.plotted_stacked_trails(quadrants),
                )

    # ========
    # Compiled results from all datasets
    # ========
    # Fit and save the total trap densities
    if args.prep_density:
        # In each image quadrant or combined quadrants
        for quadrants in quadrant_sets:
            print(
                "Fit total trap densities (%s)..." % "".join(quadrants),
                end=" ",
                flush=True,
            )
            fu.fit_total_trap_densities(dataset_list, list_name, quadrants)

    # Plot the trap density evolution
    if args.plot_density:
        print("Plot trap density evolution...", end=" ", flush=True)
        fu.plot_trap_density_evol(list_name, quadrant_sets, do_sunspots=False)
