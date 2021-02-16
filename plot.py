import matplotlib.pyplot as plt
import numpy as np


def psd(x=None, y=None, loc=None):
    if isinstance(loc, str):
        data = np.loadtxt(loc)
        x, y = data[:, 0], data[:, 1]
    elif x and y is None:
        raise AttributeError("No input given to psd. Give either x and y, or a datafile location.")

    plt.figure()
    plt.plot(x, y, linewidth=0.5)
    plt.xlabel('Frequency muHz')
    plt.ylabel('Power Spectral Density')
    plt.show()


def lc(x=None, y=None, yerr=None, loc=None, period=None):
    if isinstance(loc, str):
        data = np.loadtxt(loc)
        x, y, yerr = data[:, 0], data[:, 1], data[:, 2]
    elif x and y is None:
        raise AttributeError("No input given to psd. Give either x and y, or a datafile location.")
    plt.figure()
    plt.errorbar(x, y, yerr, fmt='k.', markersize=0.5, elinewidth=0.5)
    plt.xlabel('BJD - 2400000')
    plt.ylabel('Relative Magnitude')
    plt.ylim([np.max(y+yerr)*1.1, -0.03])

    if period is not None:
        plt.figure()
        phase = np.mod(x, period)/period
        plt.errorbar(phase, y, yerr, fmt='k.', markersize=0.5, elinewidth=0.5)
        plt.xlabel('Phase')
        plt.ylabel('Relative Magnitude')
        plt.ylim([np.max(y + yerr) * 1.1, -0.003])
    plt.show()


lc(loc='lcmag_kepler.txt', period=63.32713)
# psd(loc="datafiles/kasoc/8430105_psd.txt")
