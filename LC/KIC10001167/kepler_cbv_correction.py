from lightkurve import search_lightcurve, LightCurveCollection, read
from lightkurve.correctors import CBVCorrector
import numpy as np
from lightkurve import KeplerLightCurveFile
import matplotlib.pyplot as plt


lc_collection = search_lightcurve('KIC 10001167', author='Kepler').download_all()

# for i in range(0, len(lc_collection)):
#     lc = lc_collection[i]
#     cbvCorrector = CBVCorrector(lc)
#     cbvCorrector.correct()
#     # lc_collection[i] = cbvCorrector.corrected_lc
#     cbvCorrector.corrected_lc.to_fits(f"lc_{i}.fits", overwrite=True)
plt.figure()
ax = plt.axes()
period = 120.3903
for i in range(0, len(lc_collection)):
    lc = read(f"lc_{i}.fits")
    phase = np.mod(lc.time.value, period)/period
    plt.plot(phase, lc.normalize().flux, '.', markersize=1)
plt.show()
