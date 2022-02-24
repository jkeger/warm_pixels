import numpy as np
import pytest

from warm_pixels.pixel_lines import PixelLine, PixelLineCollection


@pytest.fixture(
    name="pixel_line"
)
def make_pixel_line():
    return PixelLine(
        background=23.3,
        data=np.array([27.3, 28, 28]),
        date=2458850.125,
        location=[14, 29],
        origin="array_A"
    )


@pytest.fixture(
    name="pixel_line_collection"
)
def make_pixel_line_collection(
        pixel_line
):
    collection = PixelLineCollection()
    collection.append([pixel_line])
    return collection


def test_locations(
        pixel_line_collection
):
    assert (pixel_line_collection.locations == np.array([[14, 29]])).all()
