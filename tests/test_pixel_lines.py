import os
from pathlib import Path

import numpy as np
import pytest

from warm_pixels.pixel_lines import PixelLine, PixelLineCollection


@pytest.fixture(
    name="save_path"
)
def make_save_path():
    path = Path(__file__).parent / "pixel_line_collection.pickle"
    yield path
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


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


def test_save_and_load(
        pixel_line_collection,
        save_path,
):
    pixel_line_collection.save(
        save_path
    )
    loaded = PixelLineCollection()
    loaded.load(save_path)
    assert (loaded.locations == np.array([[14, 29]])).all()
