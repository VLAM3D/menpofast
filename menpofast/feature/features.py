import numpy as np
from skimage.feature import daisy as skimage_daisy
from cyvlfeat.sift.dsift import dsift as cyvlfeat_dsift

from .base import ndfeature, winitfeature
from .cython import gradient as cython_gradient

scipy_gaussian_filter = None  # expensive


@ndfeature
def gradient(pixels):
    return cython_gradient(pixels)


@ndfeature
def gaussian_filter(pixels, sigma):
    global scipy_gaussian_filter
    if scipy_gaussian_filter is None:
        from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    output = np.empty(pixels.shape)
    for dim in range(pixels.shape[0]):
        scipy_gaussian_filter(pixels[dim, ...], sigma, output=output[dim, ...])
    return output


@ndfeature
def daisy(pixels, step=4, radius=15, rings=3, histograms=8, orientations=8,
          normalization='l1', sigmas=None, ring_radii=None):
    pixels = skimage_daisy(pixels[0, ...], step=step, radius=radius,
                           rings=rings, histograms=histograms,
                           orientations=orientations,
                           normalization=normalization, sigmas=sigmas,
                           ring_radii=ring_radii)

    return np.rollaxis(pixels, -1)


@winitfeature
def dsift(pixels, step=1, size=3, bounds=None, window_size=2, norm=True,
          fast=False, float_descriptors=True, geometry=(4, 4, 8)):
    
    # If norm is set to True, then the centers array will have a third column
    # with descriptor norm, or energy, before contrast normalization.
    # This information can be used to suppress low contrast descriptors.
    centers, output = cyvlfeat_dsift(
        pixels[0], step=step,
        size=size, bounds=bounds,
        norm=norm, fast=fast, float_descriptors=float_descriptors,
        geometry=geometry,
        verbose=False)

    # the output shape can be calculated from looking at the range of
    # centres / the window step size in each dimension. Note that cyvlfeat
    # returns x, y centres.
    shape = (((centers[-1, 0:2] - centers[0, 0:2]) /
              [step, step]) + 1).astype(np.int)

    # return SIFT and centers in the correct form
    return (np.require(np.rollaxis(output.reshape((shape[0], shape[1], -1)),
                                   -1),
                       dtype=np.double, requirements=['C']),
            np.require(centers.reshape((shape[0], shape[1], -1)),
                       dtype=np.int))



@ndfeature
def no_op(image_data):
    r"""
    A no operation feature - does nothing but return a copy of the pixels
    passed in.
    """
    return image_data.copy()
