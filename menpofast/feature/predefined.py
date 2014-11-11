from functools import partial

from .features import daisy, dsift


fast_daisy = partial(daisy, step=1, rings=1, radius=5, histograms=0)
fast_dsift = partial(dsift, fast=True, window_size=5, geometry=(1, 1, 8))

fast_daisy.__name__ = 'fast_daisy'
fast_dsift.__name__ = 'fast_dsift'


