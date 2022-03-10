import datetime as dt
import os
from abc import abstractmethod
from pathlib import Path

from warm_pixels import Dataset


class AbstractDatasetSource:
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def datasets(self):
        pass

    def __iter__(self):
        return iter(self.datasets())

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

    def downsample(
            self,
            start,
            step,
    ):
        return DownsampleDatasetSource(
            source=self,
            start=start,
            step=step,
        )


class FileDatasetSource(AbstractDatasetSource):
    def __init__(
            self,
            directory: Path,
            output_path: Path,
    ):
        self.directory = directory
        self.output_path = output_path

    def __str__(self):
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


class DownsampleDatasetSource(AbstractDatasetSource):
    def __init__(
            self,
            source,
            start,
            step,
    ):
        self.source = source
        self.start = start
        self.step = step

    def __str__(self):
        return f"{self.source}_downsampled_{self.start}-{self.step}"

    def datasets(self):
        return self.source.datasets()[self.start::self.step]


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

    def __str__(self):
        return f"{self.source}_{self.filter_name}"

    def datasets(self):
        return [
            dataset
            for dataset
            in self.source.datasets()
            if self.filter(dataset)
        ]
