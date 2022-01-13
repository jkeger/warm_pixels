import os

import autoarray as aa
import numpy as np
import pytest

from warm_pixels import find_warm_pixels


class TestFindWarmPixels:
    def test__find_single_pixel(self):
        # Crude test image
        image = np.ones((12, 6)) * 3
        image[5:10, 2] = [100, 10, 8, 6, 4]

        warm_pixels = find_warm_pixels(
            image=image,
            trail_length=5,
            n_parallel_overscan=0,
            n_serial_prescan=0,
            ignore_bad_columns=True,
            bad_column_factor=3.5,
            bad_column_loops=5,
            smooth_width=3,
            unsharp_masking_factor=4,
            flux_min=0,
            origin="test image",
            date="today",
        )

        assert len(warm_pixels) == 1

        line = warm_pixels[0]
        assert line.data == pytest.approx([3, 3, 3, 3, 100, 10, 8, 6, 4])
        assert line.origin == "test image"
        assert line.location == [5, 2]
        assert line.length == 9
        assert line.date == "today"
        assert line.background == pytest.approx(1, abs=1)
        assert line.flux == 100

    def test__hst_acs_example(self):
        # Load the HST ACS image
        dataset_path = os.path.join(path, "data/")
        file_path = dataset_path + "jc0a01h8q_raw.fits"
        bias_path = dataset_path + "25b1734qj_bia.fits"
        image = aa.acs.ImageACS.from_fits(
            file_path=file_path,
            quadrant_letter="A",
            bias_path=bias_path,
            bias_subtract_via_prescan=True,
        ).native

        # Extract an example patch of the full image
        row_start, row_end, column_start, column_end = -300, -100, -300, -100
        image = image[row_start:row_end, column_start:column_end]

        # Find the warm pixel trails
        warm_pixels = find_warm_pixels(image=image)

        # Result from previous version
        assert len(warm_pixels) == 1173
