# This module contains functions that are used to improve the quality of the fingerprint image.
# The enhancements consist of: normalization, segmentation, orientation, frequency and image filtering using Gabor filter.
# The input to this section is a grayscale image. The resulting image is binarized,
# the pixels representing the papillary lines have a value of 0 (black), while the background image has a value of 255 (white).

import cv2
import numpy as np
import math
import scipy.ndimage


# Image normalization
def normalize_pixel(pixel_value, mean, variance, desired_mean, desired_variance):
    """
    Function is used to normalize the image, it normalizes the value of one specific pixel.
    Function is using the formula from: https://nguyenthihanh.wordpress.com/wp-content/uploads/2015/08/handbook-of-fingerprint-recognition.pdf (strana 133).

    Args:
        pixel_value: value of actuall pixel.
        mean: mean of image.
        variance : variance of the image.
        desired_mean: desired mean of image after normalization.
        desired_variance: desired variance of image after normalization.

    Returns:
        Value of pixel after normalization.
    """

    if variance == 0:
        raise ValueError("Variance of the image cannot be zero for normalization.")

    if pixel_value > mean:
        return desired_mean + math.sqrt(
            (((pixel_value - mean) ** 2) * desired_variance) / variance
        )
    else:
        return desired_mean - math.sqrt(
            (((pixel_value - mean) ** 2) * desired_variance) / variance
        )


def normalize_image(image, desired_mean, desired_variance):
    """
    This function goes through the whole picture and normalizes every single pixel.

    Args:
        image: image which will be normalized.
        desired_mean: desired mean of image after normalization.
        desired_variance: desired  variance of image after normalization.

    Returns:
        Image after normalization.
    """
    # Exception handling
    if not isinstance(image, np.ndarray) or len(image.shape) != 2:
        raise ValueError("Image must be a 2D numpy array.")

    if not isinstance(desired_mean, int) or not isinstance(desired_variance, int):
        raise ValueError("Desired mean and variance must be integers.")

    if desired_mean < 0 or desired_variance < 0:
        raise ValueError("Desired mean and variance must be positive integers.")

    mean = np.mean(image)
    variance = np.var(image)

    # The resulting image is first created as a black image
    normalized_image = np.zeros_like(image)

    height = image.shape[0]
    width = image.shape[1]

    for x in range(0, height):
        for y in range(0, width):
            # Here the pixel value of the resulting image is calculated
            normalized_image[x, y] = normalize_pixel(
                image[x, y], mean, variance, desired_mean, desired_variance
            )

    return normalized_image


# Image segmentation
def standartize(image):
    """
    This function standartizes (normalize) the image.

    Arguments:
        image: image to be standartized.

    Returns:
        Image after standartization.
    """
    # Exception handling
    if np.std(image) == 0:
        raise ValueError("Standard deviation of input image cannot be zero.")

    return (image - np.mean(image)) / np.std(image)


def segment_image(image, block_size):
    """
    This function separates the background from the foreground of
    the input image by dividing it into several blocks.
    Then variance of each block is compared against threshold.

    Args:
        image: image to be segmented.
        block_size: size of one block.

    Returns:
        Mask of the fingerprint in the image.
    """
    # Exception handling
    if not isinstance(image, np.ndarray) or len(image.shape) != 2:
        raise ValueError("Image must be a 2D numpy array.")

    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Block size must be a positive integer greater than 0.")

    row_count = image.shape[0]
    column_count = image.shape[1]

    result = np.copy(image)
    threshold = 0.2
    threshold = np.var(image) * threshold

    # Traversing the image,
    # the first index in [] splits the panel by line,
    # the second index in [] splits the rows into a block
    for i in range(0, row_count, block_size):
        for j in range(0, column_count, block_size):
            block = image[i : i + block_size, j : j + block_size]
            block_variance = np.var(block)
            if block_variance > threshold:
                result[i : i + block_size, j : j + block_size] = np.ones_like(block)
            else:
                result[i : i + block_size, j : j + block_size] = np.zeros_like(block)

    # Mask smoothing
    contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(result, pts=[max(contours, key=cv2.contourArea)], color=(1, 1))

    kernel_size = block_size + 1 if block_size % 2 == 0 else block_size
    result = cv2.GaussianBlur(
        result, (kernel_size * 3, kernel_size * 3), kernel_size * 3
    )

    return result


# The following sections are taken from: https://github.com/cuevas1208/fingerprint_recognition.
# Docstrings and exceptions have been added to functions.
# Image orientation
def calculate_angles(im, W, smoth=False):
    """
    This function calculates the orientation of the image.

    Args:
        im: image to be processed.
        W: size of one block.
        smoth: if True, the orientation will be smoothed.

    Returns:
        Matrix (orientation field), where each element represents the orientation of one block of the input image.
    """
    # Exception handling
    if not isinstance(W, int) or W <= 0:
        raise ValueError("Block size must be a positive integer greater than 0.")

    if not isinstance(smoth, bool):
        raise ValueError("Smoth must be a boolean.")

    j1 = lambda x, y: 2 * x * y
    j2 = lambda x, y: x**2 - y**2

    (y, x) = im.shape

    # Sobel operator initialization
    sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ySobel = np.array(sobelOperator).astype(int)
    xSobel = np.transpose(ySobel).astype(int)

    result = [[] for i in range(1, y, W)]

    # Calculation of gradients
    Gx_ = cv2.filter2D(im / 125, -1, ySobel) * 125
    Gy_ = cv2.filter2D(im / 125, -1, xSobel) * 125

    # Traversing the image by blocks
    for j in range(1, y, W):
        for i in range(1, x, W):
            nominator = 0
            denominator = 0
            # Calculation of orientation in one block
            for l in range(j, min(j + W, y - 1)):
                for k in range(i, min(i + W, x - 1)):
                    Gx = round(Gx_[l, k])
                    Gy = round(Gy_[l, k])
                    nominator += j1(Gx, Gy)
                    denominator += j2(Gx, Gy)

            if nominator or denominator:
                # Orientation will be in the range 0 - pi
                angle = (math.pi + math.atan2(nominator, denominator)) / 2
                result[int((j - 1) // W)].append(angle)
            else:
                result[int((j - 1) // W)].append(0)

    result = np.array(result)

    # Orientation smoothing
    if smoth:
        result = smooth_angles(result)

    return result


def gauss(x, y):
    """
    This function calculates the value of the Gaussian function.

    Args:
        x: x coordinate.
        y: y coordinate.

    Returns:
        Value of the Gaussian function
    """
    ssigma = 1.0
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma))


def kernel_from_function(size, f):
    """
    This function creates a kernel from the given function.

    Args:
        size: size of the kernel.
        f: function which will be used to create the kernel.

    Returns:
        Kernel created from the given function (matrix).
    """
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))
    return kernel


def smooth_angles(angles):
    """
    This function smooths the orientation matrix of the image.

    Args:
        angles: matrix of the orientation of the image.

    Returns:
        Matrix of the smoothed orientation of the image.
    """
    # First, the orientation array is converted to a continuous vector array
    angles = np.array(angles)
    cos_angles = np.cos(angles.copy() * 2)
    sin_angles = np.sin(angles.copy() * 2)

    # Creating a Gaussian kernel
    kernel = np.array(kernel_from_function(5, gauss))

    # A Gaussian low-pass filter is applied to the vector field
    cos_angles = cv2.filter2D(cos_angles / 125, -1, kernel) * 125
    sin_angles = cv2.filter2D(sin_angles / 125, -1, kernel) * 125
    smooth_angles = np.arctan2(sin_angles, cos_angles) / 2

    return smooth_angles


# Image frequency
def frequest(im, orientim, kernel_size, min_wave_lenght, max_wave_lenght):
    """
    This function estimates the frequency of the fingerprint in specific block of the image.

    Args:
        im: block of the image to be processed.
        orientim: orientation of the block.
        kernel_size: size of the kernel.
        min_wave_lenght: minimal distance between two peaks in the projection of the fingerprint.
        max_wave_lenght: maximal distance between two peaks in the projection of the fingerprint.

    """
    rows, cols = np.shape(im)

    # Finding the average block orientation
    cosorient = np.cos(2 * orientim)
    sinorient = np.sin(2 * orientim)
    block_orient = math.atan2(sinorient, cosorient) / 2

    # Rotating the block so that it is perpendicular to the orientation of the papillary lines
    # i.e. the block is perpendicular to the orientation of the papillary lines
    rotim = scipy.ndimage.rotate(
        im,
        block_orient / np.pi * 180 + 90,
        axes=(1, 0),
        reshape=False,
        order=3,
        mode="nearest",
    )

    # Crop the block to contain only papillary lines
    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset : offset + cropsze][:, offset : offset + cropsze]

    # Determination of papillary line projection
    ridge_sum = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(
        ridge_sum, kernel_size, structure=np.ones(kernel_size)
    )
    ridge_noise = np.abs(dilation - ridge_sum)
    peak_thresh = 2
    maxpts = (ridge_noise < peak_thresh) & (ridge_sum > np.mean(ridge_sum))
    maxind = np.where(maxpts)  # Peaks in papillary line projection
    _, no_of_peaks = np.shape(
        maxind
    )  # Number of peaks in the projection of papillary lines

    # Determination of papillary line frequency
    # If there is only one vertex in the projection, the frequency is 0
    if no_of_peaks < 2:
        freq_block = np.zeros(im.shape)
    else:
        wave_length = (maxind[0][-1] - maxind[0][0]) / (no_of_peaks - 1)
        if wave_length >= min_wave_lenght and wave_length <= max_wave_lenght:
            freq_block = 1 / np.double(wave_length) * np.ones(im.shape)
        else:
            freq_block = np.zeros(im.shape)

    return freq_block


def ridge_freq(
    im, mask, orient, block_size, kernel_size, min_wave_lenght, max_wave_lenght
):
    """
    This function estimates the frequency of the fingerprint.

    Args:
        im: image for which the frequency will be calculated
        mask: mask of the fingerprint (after segmentation).
        orient: orientation matrix of the image.
        block_size: size of one block.
        kernel_size: size of the kernel.
        min_wave_lenght: minimal distance between two peaks in the projection of the fingerprint.
        max_wave_lenght: maximal distance between two peaks in the projection of the fingerprint.

    Returns:
        Estimated frequency of the fingerprint
    """
    # Exception handling
    if not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
        raise ValueError("Mask must be a 2D numpy array.")

    if not isinstance(orient, np.ndarray) or len(orient.shape) != 2:
        raise ValueError("Orientation must be a 2D numpy array.")

    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Block size must be a positive integer greater than 0.")

    if not isinstance(kernel_size, int) or kernel_size <= 0:
        raise ValueError("Kernel size must be a positive integer greater than 0.")

    if (
        not isinstance(min_wave_lenght, int)
        or min_wave_lenght <= 0
        or min_wave_lenght >= max_wave_lenght
    ):
        raise ValueError(
            "Minimal wave length must be a positive integer less than maximal wave length."
        )

    if not isinstance(max_wave_lenght, int) or max_wave_lenght <= 0:
        raise ValueError(
            "Maximal wave length must be a positive integer greater than 0."
        )

    rows, cols = im.shape
    freq = np.zeros((rows, cols))

    # Go through the image and calculate the frequency in each block
    for row in range(0, rows - block_size, block_size):
        for col in range(0, cols - block_size, block_size):
            # Create a block, find the orientation of the block and calculate the frequency
            image_block = im[row : row + block_size][:, col : col + block_size]
            angle_block = orient[row // block_size][col // block_size]
            if angle_block:
                freq[row : row + block_size][:, col : col + block_size] = frequest(
                    image_block,
                    angle_block,
                    kernel_size,
                    min_wave_lenght,
                    max_wave_lenght,
                )

    # Calculation of median frequency
    freq = freq * mask
    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)
    ind = np.array(ind)
    ind = ind[1, :]
    non_zero_elems_in_freq = freq_1d[0][ind]
    medianfreq = np.median(non_zero_elems_in_freq) * mask

    return medianfreq


# Image filtering using Gabor filter
def gabor_filter(im, orient, freq, kx=0.65, ky=0.65):
    """
    This function filters the image using the Gabor filter.

    Args:
        im: image to be filtered.
        orient: orientation matrix of the image.
        freq: frequency matrix of the image.
        kx: width of the Gaussian envelopes.
        ky: height of the Gaussian envelopes.

    Returns:
        Filtered image, where the fingerprint ridges are black and the background is white.
    """
    # Exception handling
    if not isinstance(orient, np.ndarray) or len(orient.shape) != 2:
        raise ValueError("Orientation must be a 2D numpy array.")

    if not isinstance(freq, np.ndarray) or len(freq.shape) != 2:
        raise ValueError("Frequency must be a 2D numpy array.")

    if not isinstance(kx, float) or not isinstance(ky, float) or kx <= 0 or ky <= 0:
        raise ValueError("Parameters of the Gaussian envelope must be positive float.")

    angle_inc = 3
    im = np.double(im)
    rows, cols = im.shape
    return_img = np.zeros((rows, cols))

    # Rounding the frequency to two decimal places to reduce the number of different frequencies
    freq_1d = freq.flatten()
    frequency_ind = np.array(np.where(freq_1d > 0))
    non_zero_elems_in_freq = freq_1d[frequency_ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100
    unfreq = np.unique(non_zero_elems_in_freq)  # Unique frequency

    # Creating a Gabor filter
    sigma_x = 1 / unfreq * kx
    sigma_y = 1 / unfreq * ky
    block_size = np.round(3 * np.max([sigma_x, sigma_y]))
    block_size = int(block_size)
    array = np.linspace(-block_size, block_size, (2 * block_size + 1))
    x, y = np.meshgrid(array, array)

    reffilter = np.exp(
        -(
            (
                (np.power(x, 2)) / (sigma_x * sigma_x)
                + (np.power(y, 2)) / (sigma_y * sigma_y)
            )
        )
    ) * np.cos(2 * np.pi * unfreq[0] * x)
    filt_rows, filt_cols = reffilter.shape
    gabor_filter = np.array(np.zeros((180 // angle_inc, filt_rows, filt_cols)))

    # Rotation of the Gabor filter
    for degree in range(0, 180 // angle_inc):
        rot_filt = scipy.ndimage.rotate(
            reffilter, -(degree * angle_inc + 90), reshape=False
        )
        gabor_filter[degree] = rot_filt

    maxorientindex = np.round(180 / angle_inc)
    # Conversion from radians to degrees
    orientindex = np.round(orient / np.pi * 180 / angle_inc)
    for i in range(0, rows // 16):
        for j in range(0, cols // 16):
            if orientindex[i][j] < 1:
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if orientindex[i][j] > maxorientindex:
                orientindex[i][j] = orientindex[i][j] - maxorientindex

    # Finding valid indexes
    block_size = int(block_size)
    valid_row, valid_col = np.where(freq > 0)
    finalind = np.where(
        (valid_row > block_size)
        & (valid_row < rows - block_size)
        & (valid_col > block_size)
        & (valid_col < cols - block_size)
    )

    # Application of Gabor filter to valid indexes
    for k in range(0, np.shape(finalind)[1]):
        r = valid_row[finalind[0][k]]
        c = valid_col[finalind[0][k]]
        img_block = im[r - block_size : r + block_size + 1][
            :, c - block_size : c + block_size + 1
        ]
        return_img[r][c] = np.sum(
            img_block * gabor_filter[int(orientindex[r // 16][c // 16]) - 1]
        )

    # Binarization of the resulting image
    # Papillary lines are black, image background is white
    gabor_img = 255 - np.array((return_img < 0) * 255).astype(np.uint8)

    return gabor_img


# Resulting improvement in image quality
def enhance_image(image_path):
    """
    This function enhances the input image using the fingerprint enhancement algorithm.

    Args:
        image_path: path to the image to be enhanced.

    Returns:
        Enhanced image.
    """
    # First the input image is loaded and converted to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Image normalization
    norm = normalize_image(image, 100, 100)
    # Image standardization
    std = standartize(image)
    # Image segmentation
    mask = segment_image(norm, 16)
    segmented_img = norm * mask
    # Orientation
    angles = calculate_angles(norm, 16, False)
    # Frequency
    freq = ridge_freq(std, mask, angles, 16, 5, 5, 15)
    # Gabor filter
    gabor = gabor_filter(std, angles, freq)

    return gabor
