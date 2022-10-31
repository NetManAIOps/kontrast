from time import time

def timer(func):
    def deco(*args, **kwargs):
        print(f'\n{func.__name__}() start running...')
        start_time = time()
        res = func(*args, **kwargs)
        end_time = time()
        print(f'{func.__name__}(): {end_time - start_time:.3f}s')
        return res
    return deco
