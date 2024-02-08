
def partial(fn, **override_kwargs):
    return lambda: lambda *args, **kwargs: fn(*args, **(kwargs.update(override_kwargs)))