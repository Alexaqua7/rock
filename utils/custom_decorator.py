from torch import no_grad

def with_torch_no_grad(func):
    def wrapper(*args, **kwargs):
        with no_grad():
            return func(*args, **kwargs)
    return wrapper