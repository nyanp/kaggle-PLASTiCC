import time
from contextlib import contextmanager


def timer(name):
    try:
        s = time.time()
        yield
    finally:
        print("[{:5g}sec] {}".format(time.time() - s, name))
