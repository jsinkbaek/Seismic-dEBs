from lightkurve import search_lightcurve, LightCurveCollection, read
from lightkurve.correctors import CBVCorrector
import numpy as np
from lightkurve import KeplerLightCurveFile
import matplotlib.pyplot as plt


lc_collection = search_lightcurve('KIC 10001167', author='Kepler').download_all()
test = CBVCorrector(lc_collection[5])
period = 120.3903
print(test.cbvs)
plt.plot(np.mod(test.cbvs[0].time.value, period)/period, test.cbvs[0].to_designmatrix().X+np.array([1-x/16 for x in range(0, 16)]))
plt.figure()
plt.plot(np.mod(lc_collection[5].time.value, period)/period, lc_collection[5].normalize().flux, 'k--')
plt.show()
test.cbvs[0].plot()
plt.show()
# for i in range(0, len(lc_collection)):
#     lc = lc_collection[i]
#     cbvCorrector = CBVCorrector(lc, )
#     cbvCorrector.correct(cbv_indices=[np.arange(1, 4)])
    # lc_collection[i] = cbvCorrector.corrected_lc
#     cbvCorrector.corrected_lc.to_fits(f"lc_{i}.fits", overwrite=True)
# plt.figure()
ax = plt.axes()
period = 120.3903
for i in range(0, len(lc_collection)):
    if i==6:
        break
    lc = read(f"lc_{i}.fits")
    phase = np.mod(lc.time.value, period)/period
    plt.plot(phase, lc.normalize().flux, '.', markersize=1)
plt.show()
