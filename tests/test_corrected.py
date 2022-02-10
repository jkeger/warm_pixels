def test_corrected(
        dataset,
        output_path,
):
    corrected = dataset.corrected()
    assert str(corrected.output_path) == f"{dataset.output_path}_corrected"
    assert len(corrected.images) == 1
