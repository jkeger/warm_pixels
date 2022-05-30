import logging
import os
from typing import List

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
            use_corrected=False,
    ):
        """
        Handles outputting data from the pipeline.

        Parameters
        ----------
        warm_pixels_
            API to access pipeline output such as warm pixels and fits
        list_name
            A name for the set of data
        use_corrected
            Are images CTI corrected?
        """
        self._warm_pixels = warm_pixels_
        self.list_name = list_name
        self.use_corrected = use_corrected

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
    pass
