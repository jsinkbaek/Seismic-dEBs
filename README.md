# Seismic-dEBs
Analysis of Eclipsing Binaries with an asteroseismic Red Giant component and a Main Sequence component.

This project is split into multiple parts.

##1: Binary_Analysis

This folder holds the inverse problem solver for use with both light curve and radial velocity data.
It is a copy of the JKTEBOP code, written by John Southworth. 
See https://www.astro.keele.ac.uk/jkt/codes/jktebop.html for details on that.

It also includes a bootstrap routine which can interface with JKTEBOP task 4 to estimate uncertainties on the found parameters.

##2: LC

A collection of scripts and function files for light curve analysis of TESS and Kepler light curves. They are specialized to the targets and datasets used, and should therefore only be
used as guidelines for analysis of other light curves than the specific ones here.



##3: RV

A spectral separation implementation following *J.F. Gonzalez and H. Levato ( A&A 448, 283-292(2006))* to calculate radial velocities of the two components using the Broadening Function Singular Value Decomposition implementation outlined
in http://www.astro.utoronto.ca/~rucinski/SVDcookbook.html. 
Should be much more general use case than the LC analysis scripts.
Also includes other general functions for spectrum analysis, such as continuum normalization, emission line reduction,
smoothing and others. Lastly, it includes some work scripts, which are tailored to specific observations, but which can also be used as examples of how to run the code.
