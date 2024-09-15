import pytest

from warm_pixels.data.source import FileDatasetSource


@pytest.fixture(name="source")
def make_source(
    dataset_list_path,
):
    return FileDatasetSource(
        dataset_list_path,
    )


def test_dataset_source(source):
    assert len(source.datasets()) == 1
    assert str(source) == "dataset_list"


@pytest.mark.parametrize(
    "date, count, name",
    [
        (6513, 1, "dataset_list_after_6513"),
        (6515, 0, "dataset_list_after_6515"),
        (6789, 0, "dataset_list_after_6789"),
    ],
)
def test_after(
    source,
    date,
    count,
    name,
):
    source = source.after(date)
    assert len(source.datasets()) == count
    assert str(source) == name


@pytest.mark.parametrize(
    "date, count, name",
    [
        (6513, 0, "dataset_list_before_6513"),
        (6515, 0, "dataset_list_before_6515"),
        (6789, 1, "dataset_list_before_6789"),
    ],
)
def test_before(
    source,
    date,
    count,
    name,
):
    source = source.before(date)
    assert len(source.datasets()) == count
    assert str(source) == name


@pytest.mark.parametrize(
    "step, count, name",
    [
        (1, 1, "dataset_list_downsampled_1"),
        (2, 1, "dataset_list_downsampled_2"),
    ],
)
def test_down_sample(
    source,
    step,
    count,
    name,
):
    source = source.downsample(
        step=step,
    )
    assert len(source.datasets()) == count
    assert str(source) == name
