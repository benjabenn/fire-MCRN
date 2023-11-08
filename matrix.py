from __future__ import annotations
import numpy as np
from cell import *


class Matrix:
    def __init__(
        self,
        cell_type_grid: np.ndarray,
        p_veg_grid: np.ndarray,
        p_den_grid: np.ndarray,
        p_wind_grid: np.ndarray,
        p_slope_grid: np.ndarray,
        p_h_grid: np.ndarray,
    ):
        if not (
            cell_type_grid.shape
            == p_veg_grid.shape
            == p_den_grid.shape
            == p_wind_grid.shape
            == p_slope_grid.shape
            == p_h_grid.shape
        ):
            raise Exception("Grids must be the same shape!")
        i: int = 0
