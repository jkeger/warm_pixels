def test_corrected(
        mock_dataset,
        output_path,
):
    corrected = mock_dataset.corrected()
    assert str(corrected.output_path) == f"{mock_dataset.output_path}_corrected"
    assert len(corrected.images) == 1
