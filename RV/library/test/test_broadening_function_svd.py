from RV.library.broadening_function_svd import *
import RV.library.test.shazam
"""
Compares broadening function implementation with already tested shazam.py
"""


def test_design_matrix():
    template_spectrum = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    span = 3
    design_matrix = DesignMatrix(template_spectrum, span)
    reference_matrix = shazam_design_matrix(template_spectrum, span)
    print('design matrix')
    print(design_matrix)
    print()
    print('shazam design matrix')
    print(reference_matrix)


def shazam_design_matrix(template_spectrum, span):
    nn = len(template_spectrum) - span + 1
    return np.matrix(-int(span/2), int(span/2+1))


def main():
    test_design_matrix()


if __name__ == "__main__":
    main()