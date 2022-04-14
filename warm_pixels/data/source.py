import datetime as dt
import os
from abc import abstractmethod
from pathlib import Path
from typing import List, Callable

from warm_pixels import Dataset


class AbstractDatasetSource:
    """
    A source of datasets. Provides a list of datasets and a name for that list.
    """

    @abstractmethod
    def __str__(self):
        """
        A name for the list of datasets
        """

    @abstractmethod
    def datasets(self):
        """
        A list of datasets
        """

    def __iter__(self):
        """
        Convenience method to iterate datasets
        """
        return iter(self.datasets())

    def __len__(self):
        return len(self.datasets())

    def after(
            self,
            date: dt.date
    ) -> "FilteredDatasetSource":
        """
        Only return datasets with an observation date after a given date.
        """
        return FilteredDatasetSource(
            f"after_{date}",
            self,
            lambda dataset: date < dataset.observation_date()
        )

    def before(
            self,
            date: dt.date
    ) -> "FilteredDatasetSource":
        """
        Only return datasets with an observation date before a given date.
        """
        return FilteredDatasetSource(
            f"before_{date}",
            self,
            lambda dataset: dataset.observation_date() < date
        )

    def downsample(
            self,
            step: int,
    ) -> "DownsampleDatasetSource":
        """
        Downsample datasets returning a every nth dataset
        """
        return DownsampleDatasetSource(
            source=self,
            step=step,
        )

    def corrected(self):
        return CorrectedDatasetSource(self)


class CorrectedDatasetSource(AbstractDatasetSource):
    def __init__(self, source):
        self.source = source

    def __str__(self):
        return f"{self.source}_corrected"

    def datasets(self):
        return [
            dataset.corrected()
            for dataset
            in self.datasets()
        ]


class FileDatasetSource(AbstractDatasetSource):
    def __init__(
            self,
            directory: Path,
            output_path: Path,
            quadrants_string: str,
    ):
        """
        Load datasets from a directory.

        Parameters
        ----------
        directory
            A directory containing directories. Each subdirectory contains one
            dataset comprising images observed at a similar time.
        output_path
            The path into which output data is put.
        """
        self.directory = directory
        self.output_path = output_path
        self.quadrants_string = quadrants_string

    def __str__(self):
        """
        The name of a dataset is simply the name of the directory that contains it
        """
        return self.directory.name

    def datasets(self) -> List[Dataset]:
        """
        A list of datasets, one for each subdirectory.
        """
        dataset_folders = os.listdir(
            self.directory
        )
        datasets = []
        for folder in dataset_folders:
            dataset = Dataset(
                self.directory / folder,
                output_path=self.output_path,
                quadrants_string=self.quadrants_string
            )
            if len(dataset) > 0:
                datasets.append(dataset)
        return sorted(
            datasets,
            key=lambda d: d.observation_date()
        )


class DownsampleDatasetSource(AbstractDatasetSource):
    def __init__(
            self,
            source: AbstractDatasetSource,
            step: int,
    ):
        """
        Downsample by returning every nth dataset.

        Parameters
        ----------
        source
            A source of datasets
        step
            The step to jump between datasets
        """
        self.source = source
        self.step = step

    def __str__(self):
        """
        The name of a downsampled dataset is the name of the source with
        _downsampled_step_size as a suffix
        """
        return f"{self.source}_downsampled_{self.step}"

    def datasets(self) -> List[Dataset]:
        """
        A list of datasets downsampled by step
        """
        return self.source.datasets()[0::self.step]


class FilteredDatasetSource(AbstractDatasetSource):
    def __init__(
            self,
            filter_name: str,
            source: AbstractDatasetSource,
            filter_: Callable,
    ):
        """
        Filter datasets.

        Parameters
        ----------
        filter_name
            The name of this filter
        source
            A source of datasets
        filter_
            A function that returns True iff the dataset passed as an
            argument should be included
        """
        self.filter_name = filter_name
        self.source = source
        self.filter = filter_

    def __str__(self):
        """
        The name of a filtered dataset comprises the name of the source
        the name of the filter
        """
        return f"{self.source}_{self.filter_name}"

    def datasets(self) -> List[Dataset]:
        """
        A list of datasets for which the filter function evaluates to True
        """
        return [
            dataset
            for dataset
            in self.source.datasets()
            if self.filter(dataset)
        ]
