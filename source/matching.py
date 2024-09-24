# This module is designed to match fingerprints using minutiae. The input is two fingerprint images,
# The output is a score that indicates how closely the two fingerprints match. The score ranges from 0 to 1,
# where 0 means that the fingerprints do not match at all, and 1 means that the fingerprints match perfectly.
# Minutiae are extracted from both prints and compared. The score is based on the number of matching minutiae.
# When two minutiae match is described in detail here: https://nguyenthihanh.wordpress.com/wp-content/uploads/2015/08/handbook-of-fingerprint-recognition.pdf (page 177).

import numpy as np
import math

import minutiaes_extraction
import image_enhancement
import binarization_and_thining


def minutiae_distance(first_minutiae_coordinates, second_minutiae_coordinates):
    """
    This function calculates the distance between two minutiae points.

    Args:
        first_minutiae: x and y coordinates of the first minutiae point.
        second_minutiae: x and y coordinates of the second minutiae point.

    Returns:
        Distance between two minutiae points.
    """
    # Exception handling
    if len(first_minutiae_coordinates) != 2:
        raise ValueError("First minutiae must be a tuple of lenght two.")

    if len(second_minutiae_coordinates) != 2:
        raise ValueError("Second minutiae must be a tuple of lenght two.")

    # Distance calculation
    dx = first_minutiae_coordinates[0] - second_minutiae_coordinates[0]
    dy = first_minutiae_coordinates[1] - second_minutiae_coordinates[1]
    distance = np.sqrt(dx * dx + dy * dy)

    return distance


def radians_to_degrees(radians):
    """
    This function converts radians to degrees.

    Args:
        radians: angle in radians.

    Returns:
        Angle in degrees.
    """

    return radians * (180 / np.pi)


def minutiae_angle_difference(first_minutiae_angle, second_minutiae_angle):
    """
    This function calculates the angle difference between two minutiae points.

    Args:
        first_minutiae: x and y coordinates of the first minutiae point.
        second_minutiae: x and y coordinates of the second minutiae point.

    Returns:
        Difference between two minutiae angles.
    """
    # Exception handling
    if not isinstance(first_minutiae_angle, float) or not isinstance(
        second_minutiae_angle, float
    ):
        raise ValueError("Angles of minutiaes have to be floats.")

    # Angle calculation
    delta_theta = abs(first_minutiae_angle - second_minutiae_angle)
    angular_dist = min(delta_theta, 360 - delta_theta)

    return angular_dist


def transform(minutiae_coordinates, change):
    """
    This function transforms the input minutiae coordinates using the calculated change parameters.

    Args:
        minutiae_coordinates: coordinates of the minutiae point.
        change: transformation parameters.

    Returns:
        New coordinates of the minutiae point.
    """
    # Exception handling
    if not isinstance(minutiae_coordinates, tuple) or len(minutiae_coordinates) != 2:
        raise ValueError("Minutiae coordinates must be a tuple of length two.")

    if not isinstance(change, tuple) or len(change) != 4:
        raise ValueError("Change must be a tuple of length four.")

    angle = math.radians(change[2])
    new_x = (
        change[3]
        * (
            minutiae_coordinates[0] * math.cos(angle)
            - minutiae_coordinates[1] * math.sin(angle)
        )
        + change[0]
    )
    new_y = (
        change[3]
        * (
            minutiae_coordinates[0] * math.sin(angle)
            + minutiae_coordinates[1] * math.cos(angle)
        )
        + change[1]
    )
    return (new_x, new_y)


# I was inspired to find out the parameters for the transformation here: https://github.com/gbnm2001/SIL775-fingerprint-matching,
# and then I modified and rewrote the code for my needs.
def get_transform_parameters(first_minutiae_set, second_minutiae_set):
    """
    This function calculates the best alignment paramaters between two sets of minutiae points using Hough Transform.

    Args:
        first_minutiae_set: first set of minutiae points.
        second_minutiae_set: second set of minutiae points.

    Returns:
        Best alignment parameters.
    """
    # Exception handling
    if not isinstance(first_minutiae_set, list) or not isinstance(
        second_minutiae_set, list
    ):
        raise ValueError("Minutiae sets must be lists.")

    if len(first_minutiae_set) == 0 or len(second_minutiae_set) == 0:
        raise ValueError("Minutiae sets must not be empty.")

    possible_transformations = {}

    # For each minutiae from the second set, a transformation is found,
    # which minutiae from the first set aligns best
    for second_minutiae in second_minutiae_set:
        for first_minutiae in first_minutiae_set:
            delta_theta = math.radians(second_minutiae[2] - first_minutiae[2])

            delta_x = (
                round(
                    second_minutiae[0]
                    - first_minutiae[0] * math.cos(delta_theta)
                    - first_minutiae[1] * math.sin(delta_theta)
                )
                // 10  # Rounding to the nearest multiple of 10
                * 10
            )
            delta_y = (
                round(
                    second_minutiae[1]
                    + first_minutiae[0] * math.sin(delta_theta)
                    - first_minutiae[1] * math.cos(delta_theta)
                )
                // 10  # Rounding to the nearest multiple of 10
                * 10
            )
            delta_theta = (
                round(math.degrees(delta_theta)) // 5 * 5
            )  # Rounding to the nearest multiple of 5
            # Every single transformation is stored in the dictionary and how many times it occurred
            if ((delta_x, delta_y, delta_theta)) in possible_transformations:
                possible_transformations[(delta_x, delta_y, delta_theta)] += 1
            else:
                possible_transformations[(delta_x, delta_y, delta_theta)] = 1

    # Selection of the most common transformations
    max_key = max(possible_transformations, key=possible_transformations.get)
    return max_key


def minutiae_matching(
    first_minutiae_set, second_minutiae_set, dist_thresh, theta_thresh, change
):
    """
    This function is calculating the matching score between two sets of minutiae points.

    Args:
        first_minutiae_set: first set of minutiae points.
        second_minutiae_set: second set of minutiae points.
        dist_thresh: distance threshold.
        theta_thresh: angle threshold.
        change: transformation parameters.

    Returns:
        Matching score between two sets of minutiae points.
    """
    # Exception handling
    # The fact that both sets of minutiae are in the required format is already taken care of
    if not dist_thresh > 0 or not theta_thresh > 0:
        raise ValueError("Distance and angle thresholds must be positive.")

    if not isinstance(change, tuple) or len(change) != 4:
        raise ValueError("Change must be a tuple of length four.")

    first_minutiae_set_lenght = len(first_minutiae_set)
    second_minutiae_set_lenght = len(second_minutiae_set)
    # Lists of already matched minutiae
    first_set_already_matched = [False for _ in first_minutiae_set]
    second_set_already_matched = [False for _ in second_minutiae_set]
    matches = []
    matched_minutiae = 0

    # Go through both sets of minutiae and find matches
    for i, minutiae_i in enumerate(first_minutiae_set):
        for j, minutiae_j in enumerate(second_minutiae_set):
            if (not second_set_already_matched[j]) and (
                not first_set_already_matched[i]
            ):
                # Transformation of coordinates and minutiae angle
                (new_x, new_y) = transform(minutiae_i[:2], change)
                new_angle = radians_to_degrees(minutiae_i[2]) + change[2]
                # Calculation of distance and angle difference
                if (
                    minutiae_distance((new_x, new_y), minutiae_j[:2]) < dist_thresh
                    and minutiae_angle_difference(
                        new_angle, radians_to_degrees(minutiae_j[2])
                    )
                    < theta_thresh
                ):
                    matched_minutiae += 1
                    matches.append((i, j))
                    first_set_already_matched[i] = True
                    second_set_already_matched[j] = True

    # A simple formula is used to calculate the score,
    # number of identical minutiae / smaller number of minutiae from two sets
    return matched_minutiae / min(first_minutiae_set_lenght, second_minutiae_set_lenght)


def hough_matching(image_path, image_path2, dist_thresh=20, theta_thresh=20):
    # Improve image quality
    first_enhancement_image = image_enhancement.enhance_image(image_path)
    second_enhancement_image = image_enhancement.enhance_image(image_path2)

    # Thinning of papillary lines
    first_thined_image = binarization_and_thining.thining(first_enhancement_image)
    second_thined_image = binarization_and_thining.thining(second_enhancement_image)

    # Getting orientation for minutiae extraction
    norm1 = image_enhancement.normalize_image(first_enhancement_image, 100, 100)
    norm2 = image_enhancement.normalize_image(second_enhancement_image, 100, 100)
    first_or = image_enhancement.calculate_angles(norm1, 16, False)
    second_or = image_enhancement.calculate_angles(norm2, 16, False)

    # Minutiae extraction
    first_img_minutiaes = minutiaes_extraction.extract_minutiaes(
        first_thined_image, first_or
    )
    second_img_minutiaes = minutiaes_extraction.extract_minutiaes(
        second_thined_image, second_or
    )

    first_complete_minutiae = first_img_minutiaes[0] + first_img_minutiaes[1]
    second_complete_minutiae = second_img_minutiaes[0] + second_img_minutiaes[1]

    delta = get_transform_parameters(first_complete_minutiae, second_complete_minutiae)
    delta = (delta[0], delta[1], delta[2], 1)
    result = minutiae_matching(
        first_complete_minutiae,
        second_complete_minutiae,
        dist_thresh,
        theta_thresh,
        delta,
    )
    return result
