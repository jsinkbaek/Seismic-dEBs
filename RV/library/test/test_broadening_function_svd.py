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
    print(design_matrix.mat)
    print()
    print('shazam design matrix')
    print(reference_matrix.T)


def shazam_design_matrix(template_spectrum, span):
    bn = span
    nn = len(template_spectrum) - span + 1
    des = np.matrix(np.zeros(shape=(bn, nn)))
    for ii in range(bn):
        des[ii,::] = template_spectrum[ii:ii+nn]

    return des


def main():
    test_design_matrix()


if __name__ == "__main__":
    main()