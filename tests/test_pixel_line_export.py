import numpy as np
import pytest

from warm_pixels import PixelLine
from warm_pixels.pixel_lines.stacked_collection import StackedPixelLine


@pytest.fixture(
    name="pixel_line"
)
def make_pixel_line():
    return PixelLine(
        location=[26, 25],
        date=20.5,
        background=23,
        flux=150,
        data=np.array([1.0, 4.0, 5.0]),
        noise=np.array([2.0, 3.0, 4.0])
    )


def test_to_dict(pixel_line):
    assert pixel_line.dict == {
        'background': 23,
        'data': [1.0, 4.0, 5.0],
        'date': 20.5,
        'flux': 150,
        'location': [26, 25],
        'noise': [2.0, 3.0, 4.0]
    }


def test_stacked_pixel_line(pixel_line):
    stacked = StackedPixelLine(
        length=3,
        background=10,
        date=1000,
        location=(0, 30),
    )
    stacked.append(pixel_line)
    stacked.append(pixel_line)

    assert stacked.dict == {
        'background': 10,
        'data': [1.0, 4.0, 5.0],
        'date': 1000,
        'flux': 5.0,
        'location': (0, 30),
        'mean_background': 23.0,
        'mean_flux': 150.0,
        'mean_row': 26.0,
        'noise': [0.0, 0.0, 0.0],
        'rms_background': 23.0,
        'rms_flux': 150.0,
        'rms_row': 26.0
    }
