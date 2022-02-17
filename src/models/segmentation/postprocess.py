import skimage.morphology as morph
from scipy import ndimage as nd


def binary_closing(image, *args, **kwargs):
    return nd.binary_closing(image, *args, **kwargs)


def binary_fill_holes(image, *args, **kwargs):
    return nd.binary_fill_holes(image, *args, **kwargs)


def erosion(image, *args, **kwargs):
    return morph.erosion(image, footprint=morph.disk(5), *args, **kwargs)


def dilation(image, *args, **kwargs):
    return morph.dilation(image, footprint=morph.disk(5), *args, **kwargs)
