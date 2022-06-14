import functools
import time


def timeRecording(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"name: {func.__qualname__},\t run time: {duration: .3f} seconds")
        return ret

    return wrapper
