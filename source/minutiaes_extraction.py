# In this module, minutiae are extracted from the fingerprint image and then checked,
# whether they are false minutiae. The Crossing Number method is used to extract the minutiae.
# This method is presented in more detail in: https://nguyenthihanh.wordpress.com/wp-content/uploads/2015/08/handbook-of-fingerprint-recognition.pdf (pages 149-150).
# To detect and remove false minutiae, the method described in: https://ieeexplore.ieee.org/document/911285 is used.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_enhancement import enhance_image, normalize_image, calculate_angles
from binarization_and_thining import thining


def extract_minutiaes(image, orientation_matrix):
    """
    This function extracts minutiaes from the input image using crossing number method.

    Args:
        image: image from which the minutiaes will be extracted.

    Returns:
        List of minutiaes, where each minutia is represented as a tuple (x, y, angle, type).
    """
    # Exception handling
    if not isinstance(image, np.ndarray) or len(image.shape) != 2:
        raise ValueError("Image must be a 2D numpy array.")

    if len(np.unique(image)) != 2:
        raise ValueError("Image must be binarized first.")

    image = cv2.bitwise_not(image)
    image = image / 255

    height = image.shape[0]
    width = image.shape[1]

    terminations = []
    bifurcations = []

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if image[i, j] == 1:

                # Calculating the crossing number for a given pixel
                crossing_number = calculate_cn(image, i, j)

                # If the crossing number is 1, check if it is a termination
                if crossing_number == 1:
                    tranzition_termination = delete_false_termination(image, i, j, 23)
                    if tranzition_termination:
                        termination_orientation = get_orientation_block(
                            i, j, 16, orientation_matrix
                        )
                        terminations.append(
                            (i, j, termination_orientation, "t")
                        )  # t = termination

                # If the crossing number is 3, check if it is a bifurcation
                elif crossing_number == 3:
                    tranzition_bifurcation = delete_false_bifurcation(image, i, j, 23)
                    if tranzition_bifurcation:
                        bifurcation_orientation = get_orientation_block(
                            i, j, 16, orientation_matrix
                        )
                        bifurcations.append(
                            (i, j, bifurcation_orientation, "b")
                        )  # b = bifurcation

    return terminations, bifurcations


def get_orientation_block(x, y, block_size, orientation_matrix):
    """
    This function returns the orientation of the block in which the pixel at the position (x, y) is located.

    Args:
        x: x coordinate of the pixel.
        y: y coordinate of the pixel.
        block_size: size of the block.
        orientation_matrix: matrix with orientation values.

    Returns:
        Orientation of the block.
    """
    block_x = x // block_size
    block_y = y // block_size

    orientation_of_block = orientation_matrix[block_x, block_y]

    return orientation_of_block


def calculate_cn(image, x, y):
    """
    This function calculates the crossing number of the pixel at the position (x, y).

    Args:
        image: image from which the crossing number will be calculated.
        x: x coordinate of the pixel.
        y: y coordinate of the pixel.

    Returns:
        Crossing number of the pixel.
    """
    # Neighborhood of the pixel to be traversed
    neighborhood = [
        image[x - 1, y - 1],
        image[x - 1, y],
        image[x - 1, y + 1],
        image[x, y + 1],
        image[x + 1, y + 1],
        image[x + 1, y],
        image[x + 1, y - 1],
        image[x, y - 1],
    ]

    # Counting crossing number
    crossing_number = 0
    for i in range(8):
        crossing_number += abs(neighborhood[i] - neighborhood[(i + 1) % 8])

    crossing_number = crossing_number / 2

    return crossing_number


def mark_the_pixels(image, x, y, minutiae_type, block_size=23):
    """
    This function creates a block around the pixel that should represent the minutiae
    from the input image. It then colors the pixels representing the papillary lines
    in this block according to the method mentioned at the beginning of the module.

    Args:
        image: image from which a block with colored pixels will be created.
        x: x coordinate of the minutiae.
        y: y coordinate of the minutiae.
        minutiae_type: type of the minutiae ('t' for termination, 'b' for bifurcation).
        block_size: size of the block around the pixel.

    Returns:
        Block with colored pixels.
    """
    # Create a copy to avoid modifying the original image
    image_copy = np.copy(image)

    # Getting a block around a pixel
    half_size = block_size // 2
    start_x = max(x - half_size, 0)
    end_x = min(x + half_size + 1, image_copy.shape[0])
    start_y = max(y - half_size, 0)
    end_y = min(y + half_size + 1, image_copy.shape[1])

    block = image_copy[start_x:end_x, start_y:end_y]
    center_x, center_y = x - start_x, y - start_y
    block[center_x, center_y] = 50

    # Directions, to traverse each pixel
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    # If it is a bifurcation, the values 100, 150, 200 are inserted into the stack
    if minutiae_type == "b":
        stack = [200, 150, 100]
        coordinates = []
        # Walk around the minutiae and gradually color the pixels in this neighborhood that represent the line
        # The coordinates of these colored pixels are also saved
        for ix, iy in directions:
            px, py = center_x + ix, center_y + iy
            if block[px, py] == 1:
                # Check if the stack is empty, which means that there was an error in detecting the minutiae
                if not len(stack):
                    return None
                value = stack.pop()
                block[px, py] = value
                coordinates.append((px, py))

        # For colored pixels, a function is called to color the entire line from the pixel to the edge of the block
        stack = [200, 150, 100]
        for ox, oy in coordinates:
            value = stack.pop()
            block = fill_one_line(block, ox, oy, value, directions)
    # If it is a termination, only one line to the edge of the block from the center of the block is colored
    else:
        block = fill_one_line(block, center_x, center_y, 100, directions)

    return block


def fill_one_line(block, x, y, value, directions):
    """
    This function colors the pixels representing the papillary line
    from the x and y coordinates to the edge of the input block.

    Args:
        block: block with the pixel to be colored.
        x: x coordinate of the pixel.
        y: y coordinate of the pixel.
        value: value to which the pixels will be colored.
        directions: directions for traversing each pixel.

    Returns:
        Block with colored one line.
    """
    queue = [(x, y)]
    while queue:
        x, y = queue.pop(0)
        if block[x, y] == 1:
            block[x, y] = value

        # Traversing neighborhood
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # If the pixel has the value 1, it is a line and is colored with the corresponding value
            if (
                0 <= nx < block.shape[0]
                and 0 <= ny < block.shape[1]
                and block[nx, ny] == 1
            ):
                block[nx, ny] = value
                queue.append(
                    (nx, ny)
                )  # Pixel is added to the queue to continue coloring

    return block


def delete_false_termination(image, x, y, block_size=23):
    """
    This function checks whether the termination minutiae is actually located at the x and y coordinates.

    Args:
        image: image from which the termination will be checked.
        x: x coordinate of the termination.
        y: y coordinate of the termination.
        block_size: size of the block around the termination.

    Returns:
        True if the termination is located at the x and y coordinates, False otherwise.
    """
    # Color the pixels in the block
    block = mark_the_pixels(image, x, y, "t", block_size)

    transition = 0

    height = block.shape[0]
    width = block.shape[1]

    # Traversing neighborhood and counting transitions between values 1 and 100
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                if block[i, j] == 100:
                    transition += 1

    # If there is only one such transition, it is indeed a minutiae termination
    return transition == 1


def delete_false_bifurcation(image, x, y, block_size=23):
    """
    This function checks whether the bifurcation minutiae is actually located at the x and y coordinates.

    Args:
        image: image from which the bifurcation will be checked.
        x: x coordinate of the bifurcation.
        y: y coordinate of the bifurcation.
        block_size: size of the block around the bifurcation.

    Returns:
        True if the bifurcation is located at the x and y coordinates, False otherwise.
    """
    # Color the pixels in the block
    block = mark_the_pixels(image, x, y, "b", block_size)

    # The case when the CN (Crossing Number) value is miscalculated,
    # it is not a bifurcation and automatically returns False
    if block is None:
        return False

    transition_100 = 0
    transition_150 = 0
    transition_200 = 0

    height = block.shape[0]
    width = block.shape[1]

    # Traversing neighborhood and counting transitions between values 1, 100, 150, and 200
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                if block[i, j] == 100:
                    transition_100 += 1
                elif block[i, j] == 150:
                    transition_150 += 1
                elif block[i, j] == 200:
                    transition_200 += 1

    # If all three transitions are one each, it is indeed a minutiae of a bifurcation
    return transition_100 == 1 and transition_150 == 1 and transition_200 == 1


def plot_minutiae(thinned_image, terminations, bifurcations):
    """
    This function plots the thinned image with marked terminations and bifurcations.

    Args:
        thinned_image: thinned image to be plotted.
        terminations: list of terminations to be marked.
        bifurcations: list of bifurcations to be marked.

    Returns:
        Plot of the thinned image with marked terminations and bifurcations.
    """
    # Exception handling
    if not isinstance(terminations, list) or not isinstance(bifurcations, list):
        raise ValueError("Terminations and bifurcations must be lists.")

    if not len(terminations) and not len(bifurcations):
        raise ValueError("No terminations or bifurcations to plot.")

    # Creating an image
    plt.figure(figsize=(9, 9))
    plt.imshow(thinned_image, cmap="gray")
    plt.axis("off")

    # First, the coordinates of the terminations are obtained and then marked in the image
    if terminations:
        x_y_pairs = [(x, y) for x, y, _, _ in terminations]
        marked_x, marked_y = zip(*x_y_pairs)
        plt.scatter(
            marked_y, marked_x, color="red", s=10, label="Terminations", alpha=0.5
        )

    # Same with the bifurcations
    if bifurcations:
        xx_yy_pairs = [(x, y) for x, y, _, _ in bifurcations]
        marked_xx, marked_yy = zip(*xx_yy_pairs)
        plt.scatter(
            marked_yy, marked_xx, color="blue", s=10, label="Bifurcations", alpha=0.5
        )

    plt.show()
