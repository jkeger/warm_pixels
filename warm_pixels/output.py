import json
import logging
import os
from typing import List

import warm_pixels.hst_utilities as ut

logger = logging.getLogger(__name__)


class OptionException(Exception):
    pass


def _check_path(
        path
):
    if path.exists():
        print(f"{path} already exists")
        return True
    os.makedirs(
        path.parent,
        exist_ok=True
    )
    return False


class AbstractOutput:
    def __init__(
            self,
            warm_pixels_,
            list_name,
    ):
        """
        Handles outputting data from the pipeline.

        Parameters
        ----------
        warm_pixels_
            API to access pipeline output such as warm pixels and fits
        list_name
            A name for the set of data
        """
        self._warm_pixels = warm_pixels_
        self.list_name = list_name

        self.all_methods = {
            name for name in dir(self)
            if not name.startswith("__")
        }

    def by_name(self, output_names: List[str]):
        """
        Output files for methods in a list of names.

        This is so plots can be conveniently passed in via the command line.

        Parameters
        ----------
        output_names
            A list of names outputs plots. These should match method names from this class
            but may use hyphens instead of underscores.

            e.g. ["density", "warm-pixel-distributions"]
        """
        for name in output_names:
            name = name.replace("-", "_")
            if name not in self.all_methods:
                raise OptionException(
                    f"{name} not a valid option. Choose from {self.all_methods}"
                )

            function = getattr(self, name)
            print(f"Outputting {name}...")
            function()
            print(f"Done.")


class Output(AbstractOutput):
    @staticmethod
    def _save_lines(pixel_lines, filename):
        with open(filename, "w+") as f:
            json.dump(
                [
                    pixel_line.dict
                    for pixel_line
                    in pixel_lines
                ],
                f
            )

    def consistent_lines(self):
        """
        Output consistent pixel lines to a JSON file
        """
        filename = ut.output_path / f"{self.list_name}_consistent.json"
        if _check_path(filename):
            return

        pixel_lines = []
        for dataset in self._warm_pixels.datasets:
            for group in dataset.groups(
                    self._warm_pixels.quadrants_string
            ):
                for quadrant in group.quadrants:
                    pixel_lines.extend(
                        quadrant.consistent_lines().lines
                    )

        self._save_lines(
            pixel_lines=pixel_lines,
            filename=filename,
        )

    def stacked_lines(self):
        """
        Output stacked pixel lines to a JSON file
        """
        filename = ut.output_path / f"{self.list_name}_stacked.json"
        if _check_path(filename):
            return

        pixel_lines = []
        for dataset in self._warm_pixels.datasets:
            for group in dataset.groups(
                    self._warm_pixels.quadrants_string
            ):
                pixel_lines.extend(
                    group.stacked_lines().lines
                )

        self._save_lines(
            pixel_lines=pixel_lines,
            filename=filename,
        )
