# This module is used to match fingerprints using markers.
# The first argument is the path to the fingerprint image we want to compare with the database, the second argument
# is the path to the database. The result is match between the input image and the database image.

import os
import argparse
import cv2

import matching


if __name__ == "__main__":
    # Get the path to the image
    parser = argparse.ArgumentParser(description="Porovnani otisku prstu s databazi")
    parser.add_argument("image_path", type=str, help="Cesta k .tif")
    parser.add_argument("db_path", type=str, help="Cesta k databazi")
    args = parser.parse_args()
    image_path = args.image_path
    db_path = args.db_path

    # Tolerance for determining the match between markers, can be optionally changed
    sd = 20
    dd = 20
    # Comparison 1:N by algorithm using minutiae
    for filename in os.listdir(db_path):
        if filename.endswith(".tif"):
            image_path2 = os.path.join(db_path, filename)
            match_score = matching.hough_matching(image_path, image_path2, 20, 20)
            print(filename)
            print(f"Best match score for input image and {filename} is: {match_score}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
