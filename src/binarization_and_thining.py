# This module is used to thin the papillary lines to one pixel width.
# The skeletonize function from the skimage library is used for this.

import numpy as np
import cv2
from skimage.morphology import skeletonize


def thining(image):
    """
    This function thins the input image using the Zhang-Suen algorithm.

    Args:
        img: binary image to be thinned.

    Returns:
        Image after thining.
    """
    # Exception handling
    if not isinstance(image, np.ndarray) or len(image.shape) != 2:
        raise ValueError("Image must be a 2D numpy array.")

    # The application of skeletonize did not show satisfactory results in the case of,
    # that the papillary lines were black and the rest of the image was white.
    # The image is therefore first inverted
    image = cv2.bitwise_not(image)

    # The skeletonize function thins the papillary lines to one pixel width
    thinned_image = skeletonize(image)

    # Convert to uint8 and convert back
    thinned_image = thinned_image.astype(np.uint8) * 255
    thinned_image = cv2.bitwise_not(thinned_image)

    return thinned_image
