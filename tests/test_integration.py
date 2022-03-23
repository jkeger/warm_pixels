import pytest

from warm_pixels import WarmPixels
from warm_pixels.plot import Plot


@pytest.mark.parametrize(
    "use_corrected",
    [False, True]
)
def test_integration(
        dataset,
        savefig_calls,
        use_corrected,
):
    if use_corrected:
        dataset = dataset.corrected()

    warm_pixels = WarmPixels(
        datasets=[dataset, dataset],
    )
    plot = Plot(
        warm_pixels_=warm_pixels,
        use_corrected=use_corrected,
        list_name="test",
    )
    plot.by_name([
        "warm-pixels",
        "warm-pixel-distributions",
        "stacked-trails",
        "density",
    ])
    assert len(savefig_calls) == 7
