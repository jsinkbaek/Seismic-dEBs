import RV.evaluate_ssr_results as evl
import matplotlib.pyplot as plt
from RV.library.linear_limbd_coeff_estimate import estimate_linear_limbd
from RV.library.initial_fit_parameters import InitialFitParameters
from RV.library.rotational_broadening_function_fitting import fitting_routine_rotational_broadening_profile as fit
from RV.library.rotational_broadening_function_fitting import get_fit_parameter_values
import os

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs')

wavelength_RV_limit = (4450, 7000)
Teff_A, Teff_B = 5042, 5621
logg_A, logg_B = 2.78, 4.58
MH_A  , MH_B   = -0.49, -0.49
mTur_A, mTur_B = 2.0, 2.0
limbd_B = estimate_linear_limbd(wavelength_RV_limit, logg_B, Teff_B, MH_B, mTur_B, loc='RV/Data/tables/atlasco.dat')

eval_data = evl.load_routine_results("RV/Data/additionals/separation_routine/", ["4500_4765", "4765_5030", "5030_5295", "5295_5560", "5560_5825", "5985_6250", "6575_6840"])
ifitpar = InitialFitParameters(vsini_guess=4.0, spectral_resolution=60000, velocity_fit_width=4.0, limbd_coef=limbd_B,
                               smooth_sigma=4.0, bf_velocity_span=300, ignore_at_phase=(0.98, 0.02))

# 4765-5030 # #
if False:
    # 9
    ifitpar.velocity_fit_width = 4.0
    RV_guess = -7.6
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[1][4][9]
    bf_smooth = eval_data.bf_results[1][6][9]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    print('9: ', RV)
    plt.show(block=True)

# # 5295-5560 # #
if True:
    # 1
    ifitpar.velocity_fit_width = 2.5
    RV_guess = -6.96
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[3][4][1]
    bf_smooth = eval_data.bf_results[3][6][1] - eval_data.bf_results[3][7][1]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    plt.show(block=False)
    print('1: ', RV)

    # 9
    ifitpar.velocity_fit_width = 3.0
    RV_guess = -6.73
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[3][4][9]
    bf_smooth = eval_data.bf_results[3][6][9]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    plt.show(block=False)
    print('9: ', RV)

    # 11
    ifitpar.velocity_fit_width = 4.0
    RV_guess = 19.0
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[3][4][11]
    bf_smooth = eval_data.bf_results[3][6][11]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    plt.show(block=False)
    print('11: ', RV)

    # 15
    RV_guess = -5.5
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[3][4][15]
    bf_smooth = eval_data.bf_results[3][6][15] - eval_data.bf_results[3][7][15]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    print('15: ', RV)
    plt.show(block=True)


# 5560-5825 Å
if False:
    # 12
    ifitpar.velocity_fit_width = 3.0
    RV_guess = -7.3
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[4][4][12]
    bf_smooth = eval_data.bf_results[4][6][12]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    print('12: ', RV)
    ifitpar.velocity_fit_width = 4.0
    plt.show(block=True)

# 5985-6250
if False:
    # 4
    ifitpar.velocity_fit_width = 3.0
    RV_guess = 21.5
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[5][4][4]
    bf_smooth = eval_data.bf_results[5][6][4]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    plt.show(block=False)
    print('4: ', RV)

    # 5
    ifitpar.velocity_fit_width = 4.0
    RV_guess = 9.25
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[5][4][5]
    bf_smooth = eval_data.bf_results[5][6][5]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    plt.show(block=False)
    print('5: ', RV)

    # 7
    ifitpar.velocity_fit_width = 4.0
    RV_guess = 17.05
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[5][4][7]
    bf_smooth = eval_data.bf_results[5][6][7]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    plt.show(block=False)
    print('7: ', RV)

    # 15
    ifitpar.velocity_fit_width = 2.5
    RV_guess = -7.3
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[5][4][15]
    bf_smooth = eval_data.bf_results[5][6][15]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    plt.show(block=False)
    print('15: ', RV)

    # 16
    ifitpar.velocity_fit_width = 3.0
    RV_guess = 12.4
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[5][4][16]
    bf_smooth = eval_data.bf_results[5][6][16]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    print('16: ', RV)
    plt.show(block=True)

# 6575-6840
if False:
    # 19
    ifitpar.velocity_fit_width = 2
    RV_guess = 6.9
    ifitpar.RV = RV_guess
    vel = eval_data.bf_results[6][4][19]
    bf_smooth = eval_data.bf_results[6][6][19]
    fit_res, model = fit(vel, bf_smooth, ifitpar, 4.0, 1.0)
    _, RV, _, _, _, _ = get_fit_parameter_values(fit_res.params)
    plt.figure()
    plt.plot(vel, bf_smooth)
    plt.plot(vel, model, 'k--')
    print('19: ', RV)
    plt.show(block=True)
