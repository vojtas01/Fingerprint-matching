# This module is used to compare fingerprints using correlation. The matchTemplate function is used for this purpose
# from the OpenCV library. The template is selected around the minutiae bifurcation closest to the center of the image. If there are no
# no minutiae, the template is selected around the center of the image. The template is then rotated by different angles and
# compared to the image. The result is the best match between the template and the image.
# Documentation for the matchTemplate function is available at: https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html

import cv2
import numpy as np

import image_enhancement
import minutiaes_extraction
import binarization_and_thining


def rotate_image(image, angle):
    """
    This function rotates the image by a given angle around its center.
    It is used to rotate the template in order to find the best match.

    Args:
        image: image to rotate.
        angle: angle to rotate the image by.

    Returns:
        Rotated image (template).
    """
    # Exception handling
    if not isinstance(image, np.ndarray) or len(image.shape) != 2:
        raise ValueError("Image must be a grayscale image")

    if not isinstance(angle, int) or angle < 0 or angle > 360:
        raise ValueError("Angle must be an integer between 0 and 360")

    center = (image.shape[1] / 2, image.shape[0] / 2)
    # Creating a rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # Apply rotation to the input image
    rotated_image = cv2.warpAffine(
        image, rotate_matrix, (image.shape[1], image.shape[0])
    )

    return rotated_image


def find_bifurcation(image, bifurcations=[], block_size=50):
    """
    This function finds the bifurcation closest to the center of the image and
    extracts a template around it.

    Args:
        image: image to search for the bifurcation.
        bifurcations: list of bifurcations in the image.
        block_size: size of the template to extract.

    Returns:
        Template around the bifurcation closest to the center.
    """
    # Exception handling
    if not isinstance(bifurcations, list):
        raise ValueError("Bifurcations must be a list of minutiaes")

    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Block size must be a positive integer")

    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2

    # If bifurcation is in the image, a template is created around them,
    # if there are no bifurcation, a template is created around the center of the image
    if bifurcations:
        coordinates = [(x, y) for x, y, _, _ in bifurcations]

        distances = np.sqrt(
            [(x - center_x) ** 2 + (y - center_y) ** 2 for (x, y) in coordinates]
        )

        # Select bifurcations that are close to the centre
        area = 150
        minutiaes_in_center = [
            bifurcations[i] for i, distance in enumerate(distances) if distance < area
        ]

        # The bifurcation closest to the centre is selected
        central_feature = minutiaes_in_center[
            np.argmin(
                [
                    distances[i]
                    for i in range(len(distances))
                    if bifurcations[i] in minutiaes_in_center
                ]
            )
        ]

        top_left_x = central_feature[1] - block_size // 2
        top_left_y = central_feature[0] - block_size // 2

        # Creating a template
        template = image[
            top_left_y : top_left_y + block_size, top_left_x : top_left_x + block_size
        ]

        return template

    else:
        start_y = center_y - block_size // 2
        start_x = center_x - block_size // 2

        template = image[start_y : start_y + block_size, start_x : start_x + block_size]

        return template


def make_template(image_path, block_size):
    """
    This function creates a template around the bifurcation closest to the center.

    Args:
        image_path: path to the image.
        block_size: size of the template to extract.

    Returns:
        Template around the bifurcation closest to the center.
    """
    # Exception handling
    if not isinstance(image_path, str):
        raise ValueError("Image path must be a string")

    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Block size must be a positive integer")

    image = image_enhancement.enhance_image(image_path)
    thined_image = binarization_and_thining.thining(image)
    norm = image_enhancement.normalize_image(image, 100, 100)
    first_or = image_enhancement.calculate_angles(norm, 16, False)
    second_img_minutiaes = minutiaes_extraction.extract_minutiaes(
        thined_image, first_or
    )
    bifurcation = second_img_minutiaes[1]
    template = find_bifurcation(image, bifurcation, block_size)

    return template
