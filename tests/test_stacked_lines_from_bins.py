def test_integration(combined_lines):
    stacked_lines = combined_lines.generate_stacked_lines_from_bins()
    assert len(stacked_lines) == 45
