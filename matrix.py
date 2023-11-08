from __future__ import annotations
import numpy as np
from cell import *

class Matrix:
    def __init__(self, cell_type_grid: np.ndarray, p_veg_grid, p_den_grid, p_wind_grid, p_slope_grid, p_h_grid) -> None:
        ...