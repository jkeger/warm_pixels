import pytest

from warm_pixels import Quadrant, QuadrantGroup


@pytest.fixture(
    name="quadrant"
)
def make_quadrant(dataset):
    return Quadrant(
        dataset=dataset,
        quadrant="A"
    )


@pytest.fixture(
    name="corrected_quadrant"
)
def make_corrected_quadrant(dataset):
    return Quadrant(
        dataset=dataset.corrected(),
        quadrant="A"
    )


def test_stacked_lines(quadrant, output_path):
    group = QuadrantGroup([quadrant])
    group.stacked_lines()

    assert (output_path / "dataset" / "A" / "stacked_lines.pickle").exists()


def test_stacked_lines_corrected(
        corrected_quadrant,
        output_path
):
    group = QuadrantGroup([corrected_quadrant])
    group.stacked_lines()

    assert (output_path / "dataset_corrected" / "A" / "stacked_lines.pickle").exists()


def test_consistent_lines(quadrant, output_path):
    quadrant.consistent_lines()

    assert (output_path / "dataset" / "A" / "consistent_lines.pickle").exists()


def test_consistent_lines_corrected(
        corrected_quadrant,
        output_path
):
    corrected_quadrant.consistent_lines()

    assert (output_path / "dataset_corrected" / "A" / "consistent_lines.pickle").exists()
