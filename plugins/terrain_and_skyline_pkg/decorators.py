import functools
import time


def decorator(func):
    """This is a generic decorator template"""
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.6f} sec.\n")
        return value
    return wrapper_timer


def debug(func):
    """This is a generic decorator template"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k} = {v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}\n")
        return value
    return wrapper_debug


def slow_down(parameter):
    """Sleep given second(s) before calling the function
    Also a template for decorator with a parameter
    """
    def slow_down_inner(func):
        @functools.wraps(func)
        def wrapper_slow_down(*args, **kwargs):
            time.sleep(parameter)
            return func(*args, **kwargs)
        return wrapper_slow_down
    return slow_down_inner


def timer_with_message(message):
    """Print the runtime of the decorated function"""
    def timer_inner(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f"{message}")
            print(f"Finished {func.__name__!r} in {run_time:.6f} sec.\n")
            return value
        return wrapper_timer
    return timer_inner