import pytest

from warm_pixels import WarmPixels, main


@pytest.mark.parametrize(
    "true_flags, n_calls",
    [
        (["prep_density", "plot_density"], 7),
        (["prep_density", "plot_density", "use_corrected"], 7),
        ([], 6),
    ]
)
def test_integration(
        dataset,
        true_flags,
        savefig_calls,
        n_calls,
):
    kwargs = {
        flag: True
        for flag
        in true_flags
    }
    overwrite = True
    warm_pixels = WarmPixels(
        datasets=[dataset, dataset],
        quadrants="A",
        overwrite=overwrite,
        list_name="test",
        plot_warm_pixels=True,
        **kwargs,
    )
    try:
        del kwargs["prep_density"]
    except KeyError:
        pass
    main(
        warm_pixels,
        list_name="test",
        plot_warm_pixels=True,
        **kwargs,
    )
    if overwrite:
        assert len(savefig_calls) == n_calls
