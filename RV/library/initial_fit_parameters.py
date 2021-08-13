
class InitialFitParameters:
    def __init__(
            self,
            vsini_guess=1.0,
            spectral_resolution=60000,
            velocity_fit_width=300,
            limbd_coef=0.68,
            smooth_sigma=4.0,
            bf_velocity_span=200,
            use_for_spectral_separation=None,
            ignore_at_phase=None
    ):
        # Value for vsini, and whether or not to fit it
        self.vsini = vsini_guess
        self.vary_vsini = True
        # Maximum change to vsini allowed each iteration in spectral_separation_routine()
        self.vsini_vary_limit = None
        # Data resolution
        self.spectral_resolution = spectral_resolution
        # How far away to include data in fitting procedure (rotational broadening function profile also masks
        # profile separately using (velocity - rv)/vsini )
        self.velocity_fit_width = velocity_fit_width
        # Linear limb darkening coefficient for rotational broadening function profile fit
        self.limbd_coef = limbd_coef
        self.vary_limbd_coef = False
        # Current RV values (used to update fit RV parameter limits correctly)
        self.RV = None
        # Smoothing value (in km/s) of the convolved gaussian used in broadening function SVD (bf_smooth()).
        self.bf_smooth_sigma = smooth_sigma
        # Width of the broadening function (in velocity space)
        self.bf_velocity_span = bf_velocity_span
        # Which spectra indices that should be used when calculating the separated component spectrum
        self.use_for_spectral_separation = use_for_spectral_separation
        # Use if component should not be subtracted in a specific phase-area (fx. (0.7, 0.9)), if it is totally eclipsed
        self.ignore_at_phase = ignore_at_phase

