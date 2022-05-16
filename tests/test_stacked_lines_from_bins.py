def _test_integration(combined_lines):
    stacked_lines = combined_lines.generate_stacked_lines_from_bins()
    assert len(stacked_lines) == 45

    line = stacked_lines[29]
    assert line.mean_background == 23.307732333125948

    data = line.data
    assert data.min() == 11.918447295319474
    assert data.max() == 229.02048409791476

    assert line.n_stacked == 2
