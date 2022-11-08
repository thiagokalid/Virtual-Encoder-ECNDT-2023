
# This class contains all parameters required in the global displacement estimation algorithm

class DisplacementParams:
    def __init__(self, method, spatial_window=None, frequency_window=None):
        # Displacement estimation method:
        if method in ["svd", "pc", None]:
            self.method = method
        else:
            raise ValueError("Specified method not implemented.")

        # Space domain window type:
        if spatial_window in ["Blackman", "Blackman-Harris", "Rectangular", None]:
            self.spatial_window = spatial_window
        else:
            raise ValueError("Specified spatial window not implemented.")

        if frequency_window in ["Stone_et_al_2001", None]:
            self.frequency_window = frequency_window
        else:
            raise ValueError("Specified frequemcy window not implemented.")
