import pytest

from warm_pixels import DatasetQuadrant, QuadrantGroup


@pytest.fixture(
    name="quadrant"
)
def make_quadrant(dataset):
    return DatasetQuadrant(
        dataset=dataset,
        quadrant="A"
    )


@pytest.fixture(
    name="corrected_quadrant"
)
def make_corrected_quadrant(dataset):
    return DatasetQuadrant(
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
