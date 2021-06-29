### Radial Velocity code for detached Eclipsing Binaries
Author: Jeppe Sinkbæk Thomsen, Master's Thesis student at Aarhus University 2021

#### Dependencies:
 - numpy      https://numpy.org/
 - barycorrpy https://pypi.org/project/barycorrpy/    (for script only)
 - astropy    https://www.astropy.org/   (for script only)
 - scipy      https://www.scipy.org/
 - matplotlib https://matplotlib.org/
 - joblib     https://joblib.readthedocs.io/en/latest/
 - lmfit      https://lmfit.github.io/lmfit-py/
 - localreg 0.3.1 https://pypi.org/project/localreg/   (for *AFS_algorithm.py* only)
 - descartes 1.1.0 https://pypi.org/project/descartes/  (for *AFS_algorithm.py* only)
 - alphashape 1.3.1 https://pypi.org/project/alphashape/ (for *AFS_algorithm.py* only)


The primary purpose of this code is radial velocity calculations for detached eclipsing binaries, specifically ones with a giant (luminous) and a main sequence (faint) component.
It should be suited for this purpose in most cases, but it would probably handle systems of closer luminosities even better regardless (it might need a few tweaks).
Additionally, it also includes some functions for processing of reduced/merged spectra in order to standardize input.

#### Library folder
Most of the actual "code" is located in the [library folder](library/). The following files present form the *core* functionality of the code:
  - [*spectral_separation_routine.py*](library/spectral_separation_routine.py). 
Includes functions that form a routine to perform spectral separation (or disentangling) and radial velocity calculation of the two components using the method described in *J.F. Gonzalez and H. Levato ( A&A 448, 283-292(2006))*.
The radial velocities are calculated using the broadening function formalism (instead of using cross correlations like described in the paper).
  - [*calculate_radial_velocities.py*](library/calculate_radial_velocities.py). Functions to calculate radial velocities from spectra using the broadening function formalism.
  - [*rotational_broadening_function_fitting.py*](library/rotational_broadening_function_fitting.py). The whole fitting routine and every function related to fitting a rotational broadening function profile to the calculated broadening function.
See *Kaluzny 2006: Eclipsing Binaries in the Open Cluster NGC 2243 II. Absolute Properties of NV CMa* for the profile used.
  - [*broadening_function_svd.py*](library/broadening_function_svd.py). The broadening function implementation following <http://www.astro.utoronto.ca/~rucinski/SVDcookbook.html> in object-oriented structure. This implementation is used as the groundwork for most of the above functions.
  - [*initial_fit_parameters.py*](library/initial_fit_parameters.py). A convenience class for storing, modifying and accessing fit parameter guesses throughout the code.

The file [*spectrum_processing_functions.py*](library/spectrum_processing_functions.py) provides a lot of general functions that are used for multiple different purposes, all relating to general work with spectroscopic data.

Additionally, the following library files provide functionality for other than core purposes:
  - [*AFS_algorithm.py*](library/AFS_algorithm.py). Provides some functions for continuum normalization of spectra by fitting to the upper boundary of an alpha shape (polygon) around the data.
Inspiration is taken from <https://ui.adsabs.harvard.edu/abs/2019AJ....157..243X/abstract>. However, the current implementation here is designed for merged spectra, not un-merged like in the paper.
  - [*linear_limbd_coeff_estimate.py*](library/linear_limbd_coeff_estimate.py). Convenience functions to estimate a linear limb darkening coefficient for a spectrum by table look-up and interpolation. Not essential, but useful as a linear limb darkening coefficient should be provided to the rotational fitting profile.

#### Data folder
The Data folder is for storing unprocessed and processed data related to the analysis. Some functions (like in *AFS_algorithm.py*) automatically saves to specific folders, which need to be available. Otherwise, most of the data saving and loading is done and designated directly in the script, so paths must be specified there. You can look at the example script for an idea of how to do it.

#### Scripts and implemenation
An example implementation of the code is provided in the script [*RV_from_spectra_kic8430105.py*](RV_from_spectra_kic8430105.py), which examines the system KIC8430105 housing an RGB star with about ~95-98% of the system luminosity, and an MS star with ~2-3% of the luminosity. 
The primary purpose of a script here is still to designate input, specifiy variables/"turn-knobs", and to act as pipeline between function calls. However, it does include some essential processing in the script, which means it is hard to call it completely separate from the code.
Therefore it is recommended to examine the example script and draw inspiration from it when trying to work with the code on a new system.
