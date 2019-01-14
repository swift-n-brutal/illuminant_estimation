
def gen_seed(i=37, n=1000000000, step=100):
    while i < n:
        yield i
        i += step
