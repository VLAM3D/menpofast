

import numpy as np

from .image import Image, MaskedImage, BooleanImage
from .image.cython import extract_patches


def convert_from_menpo(menpo_image):

    cls = eval(type(menpo_image).__name__)

    if cls is Image:
        image = cls(np.rollaxis(menpo_image.pixels, -1), copy=True)
    elif cls is MaskedImage:
        image = cls(np.rollaxis(menpo_image.pixels, -1),
                    mask=menpo_image.mask.pixels[..., 0], copy=True)
    elif cls is BooleanImage:
        image = cls(menpo_image.pixels[..., 0], copy=True)
    else:
        raise ValueError('{} is not a Menpo image class'.format(cls))

    if menpo_image.has_landmarks:
        image.landmarks = menpo_image.landmarks

    return image


def convert_to_menpo(image):

    cls = eval(type(image).__name__)

    if cls is Image:
        menpo_image = cls(np.rollaxis(image.pixels,  0, image.n_dims+1),
                          copy=True)
    elif cls is MaskedImage:
        menpo_image = cls(np.rollaxis(image.pixels, 0, image.n_dims+1),
                          mask=image.mask.pixels[0, ...], copy=True)
    elif cls is BooleanImage:
        menpo_image = cls(image.pixels[0, ...], copy=True)
    else:
        raise ValueError('{} is not a cvpr2015 image class'.format(cls))

    if image.has_landmarks:
        menpo_image.landmarks = image.landmarks

    return menpo_image


def build_parts_image(image, centres, parts_shape, offsets=np.array([[0, 0]]),
                      normalize_parts=False):

    # extract patches
    parts = extract_patches(image.pixels, np.round(centres.points),
                            np.array(parts_shape), offsets)

    # build parts image
    # img.pixels: n_centres x n_offsets x n_channels x height x width
    img = Image(parts)

    if normalize_parts:
        # normalize parts if required
        img.normalize_norm_inplace(mode='per_channel')

    return img