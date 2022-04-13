import pytest

from warm_pixels import Quadrant
from warm_pixels.model.cache import persist


@pytest.fixture(
    autouse=True
)
def patch_cache(
        monkeypatch,
        output_path
):
    monkeypatch.setattr(
        persist,
        "path",
        output_path,
    )


def test_consistent_lines(dataset, output_path):
    quadrant = Quadrant(
        dataset=dataset,
        quadrant="A"
    )

    quadrant.consistent_lines()

    assert (output_path / "dataset_A" / "consistent_lines.pickle").exists()


def test_consistent_lines_corrected(dataset, output_path):
    quadrant = Quadrant(
        dataset=dataset.corrected(),
        quadrant="A"
    )

    quadrant.consistent_lines()

    assert (output_path / "dataset_corrected_A" / "consistent_lines.pickle").exists()
