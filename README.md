# Seismic-dEBs
Analysis of Eclipsing Binaries with an asteroseismic Red Giant component and a Main Sequence component.

This project is split into two parts.

1: LC
A collection of scripts and function files for light curve analysis of TESS and Kepler light curves. They are specialized to the targets, and should therefore only be
used as guidelines for analysis of other light curves than the specific ones here.

Additionally, the folder "jktebop" holds multiple copies of the binary analysis (inverse problem solver), one for each light curve, which is used after obtaining both a reduced
light curve, and RV values for the system. 
See https://www.astro.keele.ac.uk/jkt/codes/jktebop.html for details on that.

2: RV
A spectral separation implementation to calculate radial velocities of the two components using the Broadening Function Singular Value Decomposition implementation outlined
in http://www.astro.utoronto.ca/~rucinski/SVDcookbook.html. Much more streamlined and general implementation than LC.
Also includes other general functions for spectrum analysis, such as continuum normalization (using alpha shapes, which are data encapsulating polygons), emission line reduction,
smoothing and others. Lastly, it includes some work scripts, which are tailored to specific observations, but which can also be used as examples of how to run the code.
