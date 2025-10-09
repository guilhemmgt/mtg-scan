import time
import sys

def get_elapsed_time_since(start_time : float) -> float:
    """
    Returns the elapsed time since start_time, in seconds, rounded to milliseconds.
    """
    return round(time.time() - start_time, 3)

def get_data_size(data) -> int:
    """
    Returns the size of an object, in megabytes.
    """
    return sys.getsizeof(data) / 1000000