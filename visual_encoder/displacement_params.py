# This class contains all parameters required in the global displacement estimation algorithm
import numpy as np

class DisplacementParams:
    def __init__(self, method, spatial_window=None, frequency_window=None, resolution_unit=None, xy_resolution=None,
                 rotation_correction=0):
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
        # Spatial resolution in XY plane. Currently, it is only supported mm/pixels unit.
        if resolution_unit == "mm/pixels":
            self.res_unit = resolution_unit
            if (isinstance(xy_resolution, tuple) or isinstance(xy_resolution, np.ndarray)) \
                    and len(xy_resolution) == 2:
                self.xy_res = xy_resolution
            else:
                raise ValueError("Dimension of the resolution vector not supported.")
        elif resolution_unit is None:
            self.res_unit = resolution_unit
            self.xy_res = (1, 1)  # All coordinates and shifts will be measured in pixels.
        else:
            raise ValueError("Invalid resolution unit value.")
        # Check if a rotation correction is given and is correct
        if isinstance(rotation_correction, int) or isinstance(rotation_correction, float):
            self.rot_calibration = rotation_correction
        else:
            raise ValueError("Invalid value for rotation correction")