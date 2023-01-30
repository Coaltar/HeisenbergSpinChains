from datetime import datetime


def runtime_decorator(func):
    def wrap(*args, **kwargs):
        start = datetime.now()
        res = func(*args, **kwargs)
        end = datetime.now()

        return ((end - start).total_seconds(), res)
        # print(func.__name__, ": ", end - start)

    return wrap
