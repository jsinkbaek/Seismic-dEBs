from RV.library.broadening_function_svd import *
import RV.library.test.test_library.shazam as shazam
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


def test_svd():
    template_spectrum = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    span = 3

    des_shazam = shazam_design_matrix(template_spectrum, span)
    uzam, wzam, vzam = np.linalg.svd(des_shazam.T, full_matrices=False)

    svd = SingularValueDecomposition(template_spectrum, span)
    print('uzam - svd.u')
    print(uzam- svd.u)
    print()
    print('wzam - svd.w')
    print(wzam - svd.w)
    print()
    print('vzam - svd.vH')
    print(vzam - svd.vH)
    print()
    print('svd.u')
    print(svd.u)
    print()
    print('svd.w')
    print(svd.w)
    print()
    print('svd.vH')
    print(svd.vH)


def shazam_bf(fl, tfl, bn, dv=1.0):
    bn_arr = np.arange(-int(bn / 2), int(bn / 2 + 1), dtype=float)
    vel = -bn_arr * dv

    nn = len(tfl) - bn + 1
    des = np.matrix(np.zeros(shape=(bn, nn)))
    for ii in range(bn): des[ii, ::] = tfl[ii:ii + nn]

    ## Get SVD deconvolution of the design matrix
    ## Note that the `svd` method of numpy returns
    ## v.H instead of v.
    u, w, v = np.linalg.svd(des.T, full_matrices=False)

    wlimit = 0.0
    w1 = 1.0 / w
    idx = np.where(w < wlimit)[0]
    w1[idx] = 0.0
    diag_w1 = np.diag(w1)

    vT_diagw_uT = np.dot(v.T, np.dot(diag_w1, u.T))

    ## Calculate broadening function
    bf = np.dot(vT_diagw_uT, np.matrix(fl[int(bn / 2):-int(bn / 2)]).T)
    bf = np.ravel(bf)

    return vel, bf


def shazam_smooth(vel, bf, sigma=5.0):
    nn = len(vel)
    gauss = np.zeros(nn)
    gauss[:] = np.exp(-0.5 * np.power(vel / sigma, 2))
    total = np.sum(gauss)

    gauss /= total

    bfgs = fftconvolve(bf, gauss, mode='same')

    return bfgs


def test_broadening_function():
    template_spectrum = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    program_spectrum = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    span = 3
    dv = 1.0

    vel_zam, bf_zam = shazam_bf(program_spectrum, template_spectrum, span, dv)

    BFsvd = BroadeningFunction(program_spectrum, template_spectrum, span, dv)
    BFsvd.solve()

    print('BFsvd.bf')
    print(BFsvd.bf)
    print('\nbf_zam')
    print(bf_zam)
    print('\nBFsvd.bf - bf_zam')
    print(BFsvd.bf - bf_zam)

    print('\n\nBFsvd.velocity')
    print(BFsvd.velocity)
    print('\nvel_zam')
    print(vel_zam)
    print('\nBFsvd.velocity - vel_zam')
    print(BFsvd.velocity - vel_zam)

    BFsvd.smooth_sigma = 5.0
    BFsvd.smooth()
    smooth_zam = shazam_smooth(vel_zam, bf_zam, sigma=5.0)

    print('\n\nBFsvd.bf_smooth')
    print(BFsvd.bf_smooth)
    print('\nsmooth_zam')
    print(smooth_zam)
    print('\nBFsvd.bf_smooth - smooth_zam')
    print(BFsvd.bf_smooth - smooth_zam)


def test_fit_rotational_profile():
    template_spectrum = np.array([0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    program_spectrum =  np.array([0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    span = 11
    dv = 1.0

    ifitparams = InitialFitParameters(1.0, 100, 11, 0.68)

    BFsvd = BroadeningFunction(program_spectrum, template_spectrum, span, dv)
    BFsvd.solve()
    BFsvd.smooth_sigma = 2.0
    BFsvd.smooth()
    BFsvd.fit_rotational_profile(ifitparams)

    vel_zam, bf_zam = shazam_bf(program_spectrum, template_spectrum, span, dv)
    fit_zam, model_zam, bf_smooth_zam = shazam.rotbf_fit(vel_zam, bf_zam, 11, 100, 2.0, 1.0)

    print('\n\nBFsvd.model_values')
    print(BFsvd.model_values)
    print('\nmodel_zam')
    print(model_zam)
    print('\nBFsvd.model_values - model_zam')
    print(BFsvd.model_values - model_zam)


def main():
    # test_design_matrix()
    # test_svd()
    # test_broadening_function()
    test_fit_rotational_profile()


if __name__ == "__main__":
    main()