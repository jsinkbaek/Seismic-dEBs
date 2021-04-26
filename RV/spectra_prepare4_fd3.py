import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from scipy.interpolate import interp1d
from barycorrpy import get_BC_vel, get_stellar_data, utc_tdb
from matplotlib import pyplot as plt
import RV.spectrum_processing_functions as spf

"""
This script is used to prepare observed (reduced) spectra from NOT in the form of .fits files into a format that can be
used by fd3. This process includes interpolating all the spectra to the same wavelength grid. It creates an input file
to use with the data as well, so input variables can be manually typed in this script instead.
"""

# # Set variables for script # #
data_path = 'RV/Data/unprocessed/NOT/KIC8430105/'
data_out_path = 'RV/Data/processed/NOT/KIC8430105/'
masterfile_name = 'not_kic8430105.master.obs'
observatory_location = EarthLocation.of_site("lapalma")
observatory_name = "lapalma"
stellar_target = "kic8430105"
timebase = 2400000
load_data = True       # Defines if normalized spectrum should be loaded from earlier, or done with AFS_algorithm

# #  fd3 input variables # #
ln_range = (8.0, 10.0)
out_name = 'kic8430105'
component_switches = (1, 1, 0)
noise_rms = 1.0                            # should be either a float, or np.array of shape (n, ) (n=observations)
light_factors = (0.0002855, 0.9997145)     # len=2 if component_switches (1,1,0), len=3 if (1,1,1)
                                           # can also be np.array of shape (n, len(component_switches)) if varying
independent_runs = 50
allowed_iterations_per_run = 1000
simplex_shrink_limit = 0.0001              # stops iteration if chisq fractional improvement is less than this

# wide AB--C orbit parameters (only change if component_switches=(1,1,1) and 3rd component is not static contamination)
period_abC                   = 1      # for A-B system set to 1
period_err_abC               = 0      # for A-B system set to 0
t_periastron_passage_abC     = 0      # for A-B system set to 0
t_periastron_passage_err_abC = 0      # for A-B system set to 0
eccentricity_abC             = 0
eccentricity_err_abC         = 0
periastron_longitude_abC     = 0      # for A-B system set to 0
periastron_longitude_err_abC = 0      # for A-B system set to 0
asini_ab_in_abC              = 0      # for A-B system set to 0     (a*sin(i) of AB in wide AB--C orbit)
asini_ab_err_in_abC          = 0      # for A-B system set to 0
asini_c_in_abC               = 0      # for A-B system set to 0     (a*sin(i) of C in wide AB--C orbit)
asini_c_err_in_abC           = 0      # for A-B system set to 0

# tight A--B orbit parameters
period_AB                    = 63.3271331694
period_AB_err                = 0
t_periastron_passage_AB      = 5800
t_periastron_passage_err_AB  = 5000
eccentricity_AB              = 0.2567655768
eccentricity_err_AB          = 0.05
periastron_longitude_AB      = 180
periastron_longitude_err_AB  = 180
RV_semiamplitude_A           = 43.7024471242
RV_semiamplitude_err_A       = 6
RV_semiamplitude_B           = 27.5255136967
RV_semiamplitude_err_B       = 1
perilong_advance_AB          = 0       # is not functional in current version of fd3, so should always be set to 0
perilong_advance_err_AB      = 0


# # Prepare lists and arrays # #
yvals_list = []
wl_list = []
wl_end = np.array([])
wl_start = np.array([])
delta_wl_array = np.array([])
dates = np.array([])
RA = np.array([])
DEC = np.array([])

# # Load fits files, collect and prepare data # #
for filename in os.listdir(data_path):
    if '.fits' in filename and '.lowSN' not in filename:
        with fits.open(data_path + filename) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
            wl0 = hdr['CRVAL1']
            delta_wl = hdr['CDELT1']
            date = hdr['DATE-AVG']
            vhelio = hdr['VHELIO']
            ra = hdr['OBJRA']*15.0      # degrees originally
            dec = hdr['OBJDEC']

        # Prepare before continuum fit
        filebulk = filename[:filename.rfind(".fits")]
        wl = np.linspace(wl0, wl0+delta_wl*data.size, data.size)
        selection_mask = (wl > 4200) & (wl < 9600)
        wl = wl[selection_mask]
        data = data[selection_mask]
        # Convert wl to ln(wl)
        wl = np.log(wl)
        # Do continuum fit to normalize data or load from file
        if load_data:
            load_res = np.loadtxt('RV/Data/processed/AFS_algorithm/Normalized_Spectrum/'+filebulk+'_reduced_set.dat')
            wl = load_res[:, 0]
            data = load_res[:, 1]
        else:
            wl, data, _ = spf.AFS_algorithm(wl, data, lr_frac=0.2, save_string=filebulk)
        # Cut data set down to smaller wavelength range
        selection_mask = (wl > np.log(5300)) & (wl < np.log(5700))
        wl, data = wl[selection_mask], data[selection_mask]
        # Append wavelength
        wl_list.append(wl)
        yvals_list.append(data)
        wl_end = np.append(wl_end, wl[-1])
        wl_start = np.append(wl_start, wl[0])
        dates = np.append(dates, date)
        RA = np.append(RA, ra)
        DEC = np.append(DEC, dec)

# # Plot # #
if False:
    plt.figure(figsize=(17.78, 10))
    for i in range(0, len(wl_list)):
        plt.plot(np.exp(wl_list[i]), yvals_list[i], linewidth=0.5)
    plt.xlabel('Wavelength [Å]')
    plt.show()

# # Set unified wavelength grid # #

wl0_unified = np.max(wl_start)
wl1_unified = np.min(wl_end)
wl_sizes = np.array([x.size for x in wl_list])
delta_wl_unified = (wl1_unified - wl0_unified) / np.average(wl_sizes)
wl_unified = np.arange(wl0_unified, wl1_unified, delta_wl_unified)

# # Interpolate data to wavelength grid # #
yvals_new = []
for i in range(0, len(wl_list)):
    f_intp = interp1d(wl_list[i], yvals_list[i], kind='cubic')
    yvals_new.append(f_intp(wl_unified))

# # Save spectrum data # #
save_data = np.empty(shape=(wl_unified.size, len(yvals_new)+1))
save_data[:, 0] = wl_unified

for i in range(0, len(yvals_new)):
    save_data[:, i+1] = yvals_new[i]
with open(data_out_path+masterfile_name, 'w') as f:
    f.write(f'# {save_data[0,:].size} X {save_data[:,0].size}\n')
with open(data_out_path+masterfile_name, 'ab') as f:
    f.write(b'\n')
    np.savetxt(f, save_data)


# # # Calculate Barycentric RV Correction # # #
times = Time(dates, scale='utc', location=observatory_location)  # Note: this does not correct timezone differences if
# dates are given in UTC+XX instead of UTC+00. For NOT this is probably not an issue since European Western time is
# UTC+-00. Also, I'm not sure if NOT observations are given in UTC or European Western time.
# Change to Julian Date UTC:
times.format = 'jd'
times.out_subfmt = 'long'
# print(get_stellar_data(stellar_target))    # TODO: Verify that SIMBAD results are okay
bc_rv_cor, warning, flag = get_BC_vel(times, ra=np.mean(RA), dec=np.mean(DEC), starname=stellar_target,
                                      ephemeris='de432s', obsname=observatory_name)
print()
print("RV correction")
print(bc_rv_cor)
print(warning)
print(flag)


# # # Calculate JDUTC to BJDTDB correction # # #
print()
print("Time conversion to BJDTDB")
bjdtdb, warning, flag = utc_tdb.JDUTC_to_BJDTDB(times, ra=np.mean(RA), dec=np.mean(DEC), starname=stellar_target,
                                                obsname=observatory_name)
print(bjdtdb)
print(warning)
print(flag)


# # # Create fd3 input file # # #
c_sw = component_switches
writelist = [
    f'{masterfile_name}\t{ln_range[0]} {ln_range[1]}\t{out_name}\t{c_sw[0]} {c_sw[1]} {c_sw[2]}\n',
    '\n'
]
for i in range(0, len(bjdtdb)):
    if isinstance(noise_rms, np.ndarray):
        noise_rmsi = noise_rms[i]
    else:
        noise_rmsi = noise_rms
    if isinstance(light_factors, np.ndarray):
        light_factorA = light_factors[i, 0]
        light_factorB = light_factors[i, 1]
        if light_factors[i, :].size == 3:
            light_factorC = light_factors[i, 2]
        else:
            light_factorC = ''
    else:
        light_factorA = light_factors[0]
        light_factorB = light_factors[1]
        if len(light_factors) == 3:
            light_factorC = light_factors[2]
        else:
            light_factorC = ''
    writelist.extend([
        f'{bjdtdb[i]-timebase}\t{bc_rv_cor[i]}\t{noise_rmsi}\t{light_factorA}\t{light_factorB}\t{light_factorC}\n'
    ])

writelist.extend(
    [
        '\n',
        f'{period_abC} {period_err_abC}\t{t_periastron_passage_abC} {t_periastron_passage_err_abC}\t{eccentricity_abC} '
        f'{eccentricity_err_abC}\t{periastron_longitude_abC} {periastron_longitude_err_abC}\t{asini_ab_in_abC} '
        f'{asini_ab_err_in_abC}\t{asini_c_in_abC} {asini_c_err_in_abC}\n'
    ]
)

writelist.extend(
    [
        '\n',
        f'{period_AB} {period_AB_err}\t{t_periastron_passage_AB} {t_periastron_passage_err_AB}\t{eccentricity_AB} '
        f'{eccentricity_err_AB}\t{periastron_longitude_AB} {periastron_longitude_err_AB}\t{RV_semiamplitude_A} '
        f'{RV_semiamplitude_err_A}\t{RV_semiamplitude_B} {RV_semiamplitude_err_B}\t{perilong_advance_AB} '
        f'{perilong_advance_err_AB}\n'
    ]
)

writelist.extend(
    [
        '\n',
        f'{independent_runs}\t{allowed_iterations_per_run}\t{simplex_shrink_limit}'
    ]
)
print(writelist)

with open(data_out_path+out_name+'.in', 'w') as f:
    for element in writelist:
        f.write(element)
