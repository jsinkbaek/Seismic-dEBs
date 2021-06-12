
class InitialFitParameters:
    def __init__(self, vsini_guess=1.0, spectral_resolution=60000, velocity_fit_width=300, limbd_coef=0.68):
        self.vsini = vsini_guess
        self.vary_vsini = True
        self.spectral_resolution = spectral_resolution
        self.velocity_fit_width=velocity_fit_width
        self.limbd_coef = limbd_coef
        self.vary_limbd_coef = False
        self.RV = None

