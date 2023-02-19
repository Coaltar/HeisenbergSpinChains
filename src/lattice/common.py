from datetime import datetime


def runtime_decorator(func):
    def wrap(*args, **kwargs):
        start = datetime.now()
        res = func(*args, **kwargs)
        end = datetime.now()

        return ((end - start).total_seconds(), res)
        # print(func.__name__, ": ", end - start)

    return wrap


def power_of_2(num):
    while num >= 1:
        if num == 1:
            return True
        else:
            num = num / 2
    return False
