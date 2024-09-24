# This module serves as a sample of the image_enhancement.py, binarization_and_thining.py and minutiaes_extraction.py modules.
# The argument is the path to the fingerprint image on which we want to test the functions from these three modules.
# The commented lines are used to display the individual steps of image processing. By uncommenting it, it is possible to
# show individual steps of image processing.

import argparse
import cv2

import image_enhancement
import binarization_and_thining
import minutiaes_extraction


if __name__ == "__main__":
    # Get the path to the image
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Cesta k .tif")
    args = parser.parse_args()
    image_path = args.image_path

    # Loading image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("Original image", img)

    # Enhance fingerprint image quality
    enhanced_image = image_enhancement.enhance_image(image_path)
    # cv2.imshow("Enhanced image", enhanced_image)

    # Thinning of papillary lines
    thinned_image = binarization_and_thining.thining(enhanced_image)
    # cv2.imshow("Thinned image", thinned_image)

    # Minutiae extraction
    normalized_image = image_enhancement.normalize_image(enhanced_image, 100, 100)
    orientation = image_enhancement.calculate_angles(normalized_image, 16, False)
    minutiaes = minutiaes_extraction.extract_minutiaes(thinned_image, orientation)

    # Extraction of termination and bifurcation
    terminations = minutiaes[0]
    bifurcations = minutiaes[1]

    # Plot minutiae
    # minutiaes_extraction.plot_minutiae(thinned_image, terminations, bifurcations)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
