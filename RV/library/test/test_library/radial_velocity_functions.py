import numpy as np
import os
import matplotlib.pyplot as plt


def project1_txtload(fileref):
    # Expects each file to have the same number of measurements
    with open(os.path.join('Datafiles', fileref+'.txt'), 'r') as f:
        references = []
        for line in f:
            references.append(line.split()[0])

    for i in range(0, len(references)):
        dat_temp = np.loadtxt(os.path.join('Datafiles', references[i]))
        if i == 0:
            data = np.zeros((len(references)+1, len(dat_temp[:, 1])))
            wlength = np.zeros((len(references)+1, len(dat_temp[:, 0])))

        data[i, :] = dat_temp[:, 1]
        wlength[i, :] = dat_temp[:, 0]

    # Solar reference
    data_temp = np.loadtxt(os.path.join('Datafiles', 'solar_template.txt'))
    data[-1, :] = data_temp[:, 1]
    wlength[-1, :] = data_temp[:, 0]

    data_temp = np.loadtxt(os.path.join('Datafiles', 'fileinfo.txt'))
    bjd = data_temp[:, 1]
    bar_rvc = data_temp[:, 2]
    return wlength, data, bjd, bar_rvc


def cross_correlate(a, b, spacing=None, plot=True):
    # Expects uniform and same spacing for a and b, or that spacing is calculated outside the function
    a = np.mean(a)-a
    b = np.mean(b)-b
    steps = None
    result = np.correlate(a, b, mode='same')
    result = result/np.max(result)

    if spacing is not None:
        steps = np.linspace(-len(result)//2 * spacing, len(result)//2 * spacing, len(result))
    if plot is True and spacing is not None:
        plt.plot(steps, result)
        plt.show()
    elif plot is True:
        plt.plot(result)
        plt.show()

    return result, steps


def interpol(xp, yp, x):
    y = np.zeros((len(yp[:, 0]), len(x)))
    for i in range(0, len(yp[:, 0])):
        y[i, :] = np.interp(x, xp[i, :], yp[i, :])
    return y


def phaseplot(x1, t1, x2, t2, pguess=1, repeat=True):
    """
    Creates a double star phase plot. Can be animated/interactive (meaning keyboard interaction changes it, or a simple
    plot that exits after a single loop.
    """
    plt.rcParams.update({'font.size': 30})
    loop = True
    plt.ion()
    period = pguess
    stepsize = 0.25
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot1, = ax.plot(np.mod(t1, period) / period, x1, 'b*')
    plot2, = ax.plot(np.mod(t2, period) / period, x2, 'r*')
    print('Input a to decrease period, d to increase, q to quit loop, s to halve stepsize, w to double '
          'stepsize, and p to pass (repeat same)')
    while loop is True:
        print('period', period, ', stepsize', stepsize)
        plot1.set_xdata(np.mod(t1, period) / period)
        plot2.set_xdata(np.mod(t2, period) / period)
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.legend(['Period '+'%.3f' % period + ' days'])
        ax.set_xlabel('Phase: (time modulus period) / period')
        ax.set_ylabel('Cepheid relative luminosity (V1/S1)')
        inpt = input()
        if inpt == 'q':
            loop = False
        elif inpt == 'a':
            period -= stepsize
        elif inpt == 'd':
            period += stepsize
        elif inpt == 's':
            stepsize = stepsize/2
        elif inpt == 'w':
            stepsize = stepsize*2
        elif inpt == 'p':
            pass
        if repeat is False:
            loop = False


def sine_const(x, a, freq, phase, const):
    return a*np.sin(2*np.pi*x*freq + phase) + const


def rv_func(x, k, e, freq, phase, const):
    return k * (e*np.cos(phase) + np.cos(2*np.pi*freq*x + phase)) + const
