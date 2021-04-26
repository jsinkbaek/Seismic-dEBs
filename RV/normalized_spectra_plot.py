from matplotlib import pyplot as plt; import numpy as np; import os

print(os.getcwd())
for filename in os.listdir('Data/processed/AFS_algorithm/Normalized_Spectrum/'):
    data = np.loadtxt('Data/processed/AFS_algorithm/Normalized_Spectrum/'+filename)
    plt.figure()
    plt.plot(np.exp(data[:, 0]), data[:, 1])
    plt.xlim(5300, 5700)
    plt.ylim(0.0, 1.1)
    plt.xlabel('Wavelength [Å]')
plt.show()
