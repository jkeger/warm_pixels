import pytest

from warm_pixels import WarmPixels


@pytest.mark.parametrize(
    "true_flags",
    [
        ["prep_density", "plot_density"],
        ["prep_density", "plot_density", "use_corrected"],
        [],
    ]
)
def test_integration(
        mock_dataset,
        true_flags,
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
