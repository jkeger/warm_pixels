import datetime as dt

import pytest

from warm_pixels.data.source import FileDatasetSource


@pytest.fixture(
    name="source"
)
def make_source(
        dataset_list_path,
        output_path,
):
    return FileDatasetSource(
        dataset_list_path,
        output_path,
    )


def test_dataset_source(
        source
):
    assert len(source.datasets()) == 1
    assert source.name == "dataset_list"


@pytest.mark.parametrize(
    "date, count, name",
    [
        (dt.date(2019, 12, 30), 1, "dataset_list_after_2019-12-30"),
        (dt.date(2020, 1, 1), 0, "dataset_list_after_2020-01-01"),
        (dt.date(2020, 10, 1), 0, "dataset_list_after_2020-10-01"),
    ]
)
def test_after(
        source,
        date,
        count,
        name,
):
    source = source.after(date)
    assert len(source.datasets()) == count
    assert source.name == name


@pytest.mark.parametrize(
    "date, count, name",
    [
        (dt.date(2019, 12, 30), 0, "dataset_list_before_2019-12-30"),
        (dt.date(2020, 1, 1), 0, "dataset_list_before_2020-01-01"),
        (dt.date(2020, 10, 1), 1, "dataset_list_before_2020-10-01"),
    ]
)
def test_before(
        source,
        date,
        count,
        name,
):
    source = source.before(date)
    assert len(source.datasets()) == count
    assert source.name == name
