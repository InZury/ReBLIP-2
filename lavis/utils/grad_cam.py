# Some portions of this file were inspired by code from [LAVIS].
# The original code is distributed under the BSD 3-Clause License, with the following copyright notice:
# Copyright (c) [2023], [salesforce.com, inc.]

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import filters
from skimage import transform as skimage_transform


def get_activation_map(image, activation_map, blur=True, overlap=True):
    activation_map -= activation_map.min()

    if activation_map.max() > 0:
        activation_map /= activation_map.max()

    activation_map = skimage_transform.resize(activation_map, (image.shape[:2]), order=3, mode="constant")

    if blur:
        activation_map = filters.gaussian_filter(activation_map, 0.02 * max(image.shape[:2]))
        activation_map -= activation_map.min()
        activation_map /= activation_map.max()

    cmap = plt.get_cmap("jet")
    activation_map_vector = cmap(activation_map)
    activation_map_vector = np.delete(activation_map_vector, 3, 2)

    if overlap:
        activation_map = (
            1 * (1 - activation_map**0.7).reshape(activation_map.shape + (1, )) * image
            + (activation_map**0.7).reshape(activation_map.shape + (1, )) * activation_map_vector
        )

    return activation_map
