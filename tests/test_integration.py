from warm_pixels import WarmPixels, output_plots


def test_integration(
        dataset,
        savefig_calls,
):
    warm_pixels = WarmPixels(
        datasets=[dataset, dataset],
        quadrants="A",
    )
    output_plots(
        warm_pixels,
        list_name="test",
        plot_warm_pixels=True,
        plot_density=True,
    )
    assert len(savefig_calls) == 7


def test_integration_corrected(
        dataset,
        savefig_calls,
):
    dataset = dataset.corrected()
    warm_pixels = WarmPixels(
        datasets=[dataset, dataset],
        quadrants="A",
    )
    output_plots(
        warm_pixels,
        list_name="test",
        plot_warm_pixels=True,
        plot_density=True,
    )
    assert len(savefig_calls) == 7
