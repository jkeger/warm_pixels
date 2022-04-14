def test_corrected(
        dataset,
        output_path,
):
    corrected = dataset.corrected()
    assert corrected.path == dataset.path / "corrected"
    assert len(corrected.images) == 1
