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
    return PixelLineCollection([pixel_line])


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
    loaded = PixelLineCollection.load(save_path)
    assert (loaded.locations == np.array([[14, 29]])).all()


def test_equality(pixel_line):
    assert PixelLineCollection([
        pixel_line
    ]) == PixelLineCollection([
        pixel_line
    ])


def test_add(
        pixel_line,
        pixel_line_collection
):
    combined = pixel_line_collection + pixel_line_collection
    assert isinstance(combined, PixelLineCollection)
    assert (
            PixelLineCollection([
                pixel_line, pixel_line
            ]) == combined
    ).all()


def test_sum(
        pixel_line_collection,
        pixel_line,
):
    combined = sum(pixel_line_collection, pixel_line_collection)
    assert isinstance(combined, PixelLineCollection)
    assert (
            PixelLineCollection([
                pixel_line, pixel_line
            ]) == combined
    ).all()
