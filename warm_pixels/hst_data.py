"""Defines where the HST data is located on disc, and a class to read it in/contain it"""

from glob import glob
from pathlib import Path

import autoarray as aa
import numpy as np
from autoarray.instruments.acs import ImageACS

from warm_pixels import hst_utilities as ut


class Image:
    def __init__(
            self,
            path: Path
    ):
        self.path = path

    @property
    def name(self):
        return self.path.name.split("_")[0]

    @property
    def cor_path(self):
        return self.path.parent / f"{self.name}_raw_cor.fits"

    def image(self):
        return aa.acs.ImageACS.from_fits(
            file_path=str(self.path),
            quadrant_letter="A"
        )

    def load_quadrant(self, quadrant):
        return ImageACS.from_fits(
            file_path=str(self.path),
            quadrant_letter=quadrant,
            bias_subtract_via_bias_file=True,
            bias_subtract_via_prescan=True,
        ).native

    def date(self):
        return 2400000.5 + self.image().header.modified_julian_date


class Dataset:
    def __init__(
            self,
            path: Path
    ):
        """Simple class to store a list of image file paths and mild metadata.

        Parameters
        ----------
        path
            The path to a directory containing fits files

        Attributes
        ----------
        path : str
            File path to the dataset directory.
        """
        self.path = path

    def __len__(self):
        return len(self.images)

    @property
    def images(self):
        return [
            Image(
                self.path / name,
            )
            for name in glob(
                f"{self.path}/*_raw.fits"
            )
        ]

    @property
    def name(self):
        return self.path.name

    @property
    def date(self):
        """Return the Julian date of the set, taken from the first image."""
        return self.images[0].date()

    # ========
    # File paths for saved data, including the quadrant(s)
    # ========
    def saved_lines(self, quadrant):
        """Return the file name including the path for saving derived data."""
        return self.path / f"saved_lines_{quadrant}.pickle"

    def saved_consistent_lines(self, quadrant, use_corrected=False):
        """Return the file name including the path for saving derived data."""
        suffix = "_cor" if use_corrected else ""
        return self.path / f"saved_consistent_lines_{quadrant}{suffix}.pickle"

    def saved_stacked_lines(self, quadrants, use_corrected=False):
        """Return the file name including the path for saving derived data."""
        suffix = "_cor" if use_corrected else ""
        quadrant_string = "".join(quadrants)
        return self.path / f"saved_stacked_lines_{quadrant_string}{suffix}.pickle"

    def saved_stacked_info(self, quadrants, use_corrected=False):
        """Return the file name including the path for saving derived data."""
        suffix = "_cor" if use_corrected else ""
        quadrant_string = "".join(quadrants)
        return self.path / f"saved_stacked_info_{quadrant_string}{suffix}.npz"

    def plotted_stacked_trails(self, quadrants, use_corrected=False):
        """Return the file name including the path for saving derived data."""
        suffix = "_cor" if use_corrected else ""
        return ut.path + "/stacked_trail_plots/%s_plotted_stacked_trails_%s%s.png" % (
            self.name,
            "".join(quadrants),
            suffix,
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
# datasets_test = ["huff_spt814b"]
datasets_test = ["07_2020"]
datasets_test_2 = [
    "richmassey60491",  # 2006/08/16, day 1629, 7 images
    "richmassey61092",  # 2006/09/11, day 1655, 4 images
    "04_2017",  # 2017/04/05, day 5514, 7 images
    "10_2017",  # 2017/10/03, day 5695, 5 images
]
datasets_test_3 = [
    "shortSNe2",  # 2003/05/06, day 431, 3 images
]

# Dictionary of choosable list names
dataset_options = {
    # "all": datasets_all,
    "sample": datasets_sample,
    "test": datasets_test,
    "test_2": datasets_test_2,
    "test_3": datasets_test_3,
}

dataset_path = Path(
    __file__
).parent.parent.parent / "hst_acs_datasets"

# Convert all to Dataset objects

dataset_lists = {
    key: [
        Dataset(
            dataset_path / dataset
        ) for dataset
        in dataset_options[key]
    ]
    for key
    in dataset_options.keys()
}
