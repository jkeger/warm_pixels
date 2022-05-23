import bz2
import functools
import os
import pickle
from functools import wraps
from pathlib import Path

from warm_pixels import hst_utilities as hu


def cache(func):
    """
    Cache the results of a method that takes no arguments
    """

    @wraps(func)
    def wrapper(*args):
        self = args[0]
        key = f"__{func.__name__}"
        if key not in self.__dict__:
            self.__dict__[key] = func(self)
        return self.__dict__[key]

    return wrapper


class Persist:
    def __init__(self, path: Path):
        """
        Save the output of methods and load it rather than
        calling the method to avoid re-executing expensive
        steps in the pipeline.

        Parameters
        ----------
        path
            A path in which cached data is saved.
        """
        self.path = path

    def __call__(self, directory_func=str):
        """
        Create a decorator with a function to convert an object into
        a directory name

        Parameters
        ----------
        directory_func
            A function that converts an instance into a directory

        Returns
        -------

        """

        def outer(func):
            """
            Decorate the function to persist output.

            Files are saved in a directory named after str(instance); each
            instance that uses Persist must implement __str__.

            Files are saved with the name of the function.

            Parameters
            ----------
            func
                Some method of a class for which data is saved and loaded to
                avoid calling the method.

            Returns
            -------
            A decorated function.
            """
            func = cache(func)

            name = f"{func.__name__}.pbz2"

            @functools.wraps(func)
            def wrapper(instance):
                directory = self.path / directory_func(instance)
                path = directory / name
                os.makedirs(directory, exist_ok=True)

                if path.exists():
                    with bz2.BZ2File(path) as f:
                        return pickle.load(f)

                result = func(instance)
                with bz2.BZ2File(path, "w") as f:
                    pickle.dump(result, f)

                return result

            return wrapper

        return outer


persist = Persist(hu.cache_path)
