import pytest

import warm_pixels as wp


@pytest.fixture(
    name="quadrant"
)
def make_quadrant(dataset):
    return wp.Quadrant(
        dataset=dataset,
        quadrant="A"
    )


def test_warm_pixels_directory_function(dataset):
    assert wp.directory_func(
        wp.WarmPixels([dataset])
    ) == "380a702f2184cc52bc1fadd47fb07f44"


def test_stacked_lines(quadrant, output_path):
    group = wp.QuadrantGroup([quadrant])
    group.stacked_lines()

    assert (output_path / "dataset" / "A" / "stacked_lines.pickle").exists()


def test_consistent_lines(quadrant, output_path):
    quadrant.consistent_lines()

    assert (output_path / "dataset" / "A" / "consistent_lines.pickle").exists()


def test_consistent_lines_corrected(dataset, output_path):
    quadrant = wp.Quadrant(
        dataset=dataset.corrected(),
        quadrant="A"
    )

    quadrant.consistent_lines()

    assert (output_path / "dataset_corrected" / "A" / "consistent_lines.pickle").exists()
