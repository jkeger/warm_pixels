import pytest

from warm_pixels import WarmPixels


@pytest.mark.parametrize(
    "prep_density",
    [False, True]
)
@pytest.mark.parametrize(
    "use_corrected",
    [False, True]
)
@pytest.mark.parametrize(
    "plot_density",
    [False, True]
)
def test_integration(
        mock_dataset,
        prep_density,
        use_corrected,
        plot_density,
):
    WarmPixels(
        datasets=[mock_dataset],
        quadrants="A",
        overwrite=True,
        prep_density=prep_density,
        use_corrected=use_corrected,
        plot_density=plot_density,
    ).main()
