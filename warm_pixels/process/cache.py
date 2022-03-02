from functools import wraps


def cache(func):
    @wraps(func)
    def wrapper(*args):
        self = args[0]
        key = f"__{func.__name__}"
        if key not in self.__dict__:
            self.__dict__[key] = func(self)
        return self.__dict__[key]

    return wrapper
