# Fingerprint matching algorithms

## Description

This repository offers an implementation of two fingerprint matching algorithms. 

- The first one uses the minutiae matching method (characteristic points that can be found on fingerprints). 
- The second one uses correlation approach.

The repository also includes modules that are used to preprocess fingerprint images. These modules improve the quality of the fingerprint image, highlight its texture and search for minutiae in the fingerprint. 

Both algorithms were programmed as practical part of my Bachelor's thesis. The repository also contains two datasets that can be used to test both algorithms. Each contains 80 fingerprint images (10 different fingerprints, each captured 8 times) in .tif format.


## Prerequisites

1. Python version >= 3.12.3.

2. [pip](https://pip.pypa.io/en/stable/installation/).

3. Requirements from the attached file that can be installed as follows:
    ```
    $ pip install -r requirements.txt
    ```


## Showcase modules to try out

Three modules were created to test out both algorithms:

1. enhancement_example.py
2. matching_example.py
3. example_correlation.py

### enhancement_example.py

Using this module, it is possible to test fingerprint image quality enhancement, fingerprint ridge thinning and minutiae detection. 

You can run this module using: 
```
$ python enhancement_example.py <image_path>
```
- <strong> <image_path> </strong> - path to the .tif image you want to test, e.g. from the attached database

### matching_example.py

This module can be used to test an algorithm that matches fingerprints using minutiae. You can run this module using:
```
$ python matching_example.py <image_path> <database_path>
```
- <strong> <image_path> </strong> - path to the .tif image you want to test
- <strong> <database_path> </strong> - path to the database.

<i> The module compares the image (<image_path>) with all the images in the database (<database_path>). After each comparison, the similarity score between the input image and the database image is printed. </i>


### example_correlation.py

The example_correlation.py module can be used to test an algorithm that compares fingerprints using correlation. You can run the module using:
```
$ python example_correlation.py <image_path> <database_path> 
```
- <strong> <image_path> </strong> - path to the .tif image you want to test
- <strong> <database_path> </strong> - path to the database

This module works in the same way as the matching_example.py module.

## References

- [Handbook of Fingerprint Recognition, Davide Maltoni, Dario Maio, Anil K. Jain, Salil Prabhakar](https://nguyenthihanh.wordpress.com/wp-content/uploads/2015/08/handbook-of-fingerprint-recognition.pdf)

- [An algorithm for fingerprint image postprocessing, Marius Tico, Pauli Kuosmanen](https://ieeexplore.ieee.org/document/911285)

- [Fingerprint image enhancement: algorithm and performance evaluation, Lin Hong, Yifei Wan, Anil Jain](https://ieeexplore.ieee.org/document/709565)

- [Fingerprint Image Enhancement and Minutiae Extraction, Raymond Thai](https://www.peterkovesi.com/studentprojects/raymondthai/RaymondThai.pdf)

- [Fingerprint recognition by Manuel Cuevas (GitHub Repository)](https://github.com/cuevas1208/fingerprint_recognition?tab=readme-ov-file)

- [Fingerprint matching by Gautam Meeshi (GitHub Repository)](https://github.com/gbnm2001/SIL775-fingerprint-matching)


Any bug reports or improvements are welcome, please contact me at: vojtasedl@seznam.cz
