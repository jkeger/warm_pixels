def test_corrected(
        dataset,
        output_path,
):
    corrected = dataset.corrected()
    assert corrected.output_path == dataset.output_path / "corrected"
    assert len(corrected.images) == 1
