# This module is used to test the fingerprint matching algorithm using correlation.
# The first argument is the path to the fingerprint image we want to compare with the database.
# The second argument is the path to the database. The result is the match between the input image and the database image.

import argparse
import cv2
import os

import image_enhancement
import correlation

if __name__ == "__main__":
    # Get the path to the image
    parser = argparse.ArgumentParser(description="Porovnani otisku prstu s databazi")
    parser.add_argument("image_path", type=str, help="Cesta k .tif")
    parser.add_argument("db_path", type=str, help="Cesta k databazi")
    args = parser.parse_args()
    image_path = args.image_path
    db_path = args.db_path

    # Create template for comparison
    template_size = 50
    template = correlation.make_template(image_path, template_size)

    # Comparing 1:N using a correlation algorithm
    for filename in os.listdir(db_path):
        if filename.endswith(".tif"):
            image_path2 = os.path.join(db_path, filename)
            image = image_enhancement.enhance_image(image_path2)

            print(filename)
            best_score = 0
            best_loc = None

            # The template is rotated by 10 degrees and compared with the image
            for angle in range(0, 360, 10):
                rotated_template = correlation.rotate_image(template, angle)
                result = cv2.matchTemplate(
                    image, rotated_template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_score:
                    best_score = max_val
                    best_loc = max_loc

            print(f"Best match score for input image and {filename} is: {best_score}")
