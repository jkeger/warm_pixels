from warm_pixels import Quadrant


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
