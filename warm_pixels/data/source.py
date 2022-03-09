import datetime as dt
import os
from abc import abstractmethod
from pathlib import Path

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
