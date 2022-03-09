import datetime as dt
import os
from abc import abstractmethod
from pathlib import Path

import pytest

from warm_pixels import Dataset


class AbstractDatasetSource:
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def datasets(self):
        pass

    def after(self, date: dt.date):
        return FilteredDatasetSource(
            f"after_{date}",
            self,
            lambda dataset: date < dataset.observation_date()
        )

    def before(self, date: dt.date):
        return FilteredDatasetSource(
            f"before_{date}",
            self,
            lambda dataset: dataset.observation_date() < date
        )


class FileDatasetSource(AbstractDatasetSource):
    def __init__(
            self,
            directory: Path,
            output_path: Path,
    ):
        self.directory = directory
        self.output_path = output_path

    @property
    def name(self):
        return self.directory.name

    def datasets(self):
        dataset_folders = os.listdir(
            self.directory
        )
        return [
            Dataset(
                self.directory / folder,
                output_path=self.output_path
            )
            for folder in dataset_folders
        ]


class FilteredDatasetSource(AbstractDatasetSource):
    def __init__(
            self,
            filter_name,
            source,
            filter_
    ):
        self.filter_name = filter_name
        self.source = source
        self.filter = filter_

    @property
    def name(self):
        return f"{self.source.name}_{self.filter_name}"

    def datasets(self):
        return [
            dataset
            for dataset
            in self.source.datasets()
            if self.filter(dataset)
        ]


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
