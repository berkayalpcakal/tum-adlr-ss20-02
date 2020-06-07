import functools


def print_message(msg: str):
    def decorator(func):
        @functools.wraps(func)
        def printer(*args, **kwargs):
            print(msg, end="")
            res = func(*args, **kwargs)
            print("[DONE]")
            return res
        return printer
    return decorator
