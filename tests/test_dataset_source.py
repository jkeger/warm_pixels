import datetime as dt
import os
from abc import abstractmethod
from pathlib import Path

import pytest

from warm_pixels import Dataset


class AbstractDatasetSource:
    @abstractmethod
    def datasets(self):
        pass

    def after(self, date: dt.date):
        return FilteredDatasetSource(
            self,
            lambda dataset: date < dataset.observation_date()
        )


class FileDatasetSource(AbstractDatasetSource):
    def __init__(
            self,
            directory: Path,
            output_path: Path,
    ):
        self.directory = directory
        self.output_path = output_path

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
            source,
            *filters
    ):
        self.source = source
        self.filters = filters

    def datasets(self):
        return [
            dataset
            for dataset
            in self.source.datasets()
            if all(
                filter_(dataset)
                for filter_
                in self.filters
            )
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


def test_filter(
        source
):
    source = source.after(
        dt.date(2019, 12, 30)
    )
    assert len(source.datasets()) == 1
