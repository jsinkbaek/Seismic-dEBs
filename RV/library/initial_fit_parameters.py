
class InitialFitParameters:
    def __init__(self, vsini_guess=1.0, spectral_resolution=60000, velocity_fit_width=300, limbd_coef=0.68,
                 smooth_sigma=4.0):
        self.vsini = vsini_guess
        self.vary_vsini = True
        self.vsini_vary_limit = None
        self.spectral_resolution = spectral_resolution
        self.velocity_fit_width = velocity_fit_width
        self.limbd_coef = limbd_coef
        self.vary_limbd_coef = False
        self.RV = None
        self.bf_smooth_sigma = smooth_sigma

