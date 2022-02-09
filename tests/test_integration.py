import pytest

from warm_pixels import WarmPixels


@pytest.mark.parametrize(
    "true_flags, n_calls",
    [
        (["prep_density", "plot_density"], 7),
        (["prep_density", "plot_density", "use_corrected"], 5),
        ([], 6),
    ]
)
def test_integration(
        mock_dataset,
        true_flags,
        savefig_calls,
        n_calls,
):
    kwargs = {
        flag: True
        for flag
        in true_flags
    }
    WarmPixels(
        datasets=[mock_dataset, mock_dataset],
        quadrants="A",
        overwrite=True,
        **kwargs,
    ).main()
    assert len(savefig_calls) == n_calls
