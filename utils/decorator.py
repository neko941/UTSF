def _list(func):
    def wrap(a):
        if isinstance(a, list): retval = func(a)
        elif isinstance(a, tuple): retval = func(list(a))
        else: retval = func([a])
        return retval
    return wrap