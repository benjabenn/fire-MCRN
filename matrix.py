from __future__ import annotations
import numpy as np
from cell import *


class Matrix:
    cell_matrix: np.ndarray

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
        self.cell_matrix = np.empty(shape=cell_type_grid.shape, dtype=Cell)
        for i in range(self.cell_matrix.shape[1]):
            for j in range(self.cell_matrix.shape[0]):
                if cell_type_grid[i][j] == 1:
                    self.cell_matrix[i][j] = NoFuelState(CellProbabilities(
                        p_veg_grid[i][j], p_den_grid[i][j], p_wind_grid[i][j], p_slope_grid[i][j], p_h_grid[i][j]))
                elif cell_type_grid[i][j] == 2:
                    self.cell_matrix[i][j] = FuelState(CellProbabilities(
                        p_veg_grid[i][j], p_den_grid[i][j], p_wind_grid[i][j], p_slope_grid[i][j], p_h_grid[i][j]))
                elif cell_type_grid[i][j] == 3:
                    self.cell_matrix[i][j] = BurningState(CellProbabilities(
                        p_veg_grid[i][j], p_den_grid[i][j], p_wind_grid[i][j], p_slope_grid[i][j], p_h_grid[i][j]))
                elif cell_type_grid[i][j] == 4:
                    self.cell_matrix[i][j] = BurntState(CellProbabilities(
                        p_veg_grid[i][j], p_den_grid[i][j], p_wind_grid[i][j], p_slope_grid[i][j], p_h_grid[i][j]))
    
    def get_cell(self, row: int, col: int) -> Cell:
        return self.cell_matrix[row][col]
    
    def get_neighbors(self, row: int, col: int, radius: int = 1) -> list[Cell]:
        rows: int = self.cell_matrix.shape[1]
        cols: int = self.cell_matrix.shape[0]
        neighbors: list[Cell] = []

        for i in range(row - radius, row + radius + 1):
            for j in range(col - radius, col + radius + 1):
                if 0 <= i < rows and 0 <= j < cols and (i != row or j != col):
                    neighbors.append(self.cell_matrix[i][j])

        return neighbors
    
    def update(self):
        new_cell_matrix: np.ndarray = np.empty_like(self.cell_matrix) 
        for i in range(self.cell_matrix.shape[1]):
            for j in range(self.cell_matrix.shape[0]):
                new_cell_matrix[i][j] = self.cell_matrix[i][j].apply_rules(self.get_neighbors(i, j))
        self.cell_matrix = new_cell_matrix

    def render_emojis(self):
        emoji_matrix = np.empty_like(self.cell_matrix, dtype=object)
        for i in range(self.cell_matrix.shape[1]):
            for j in range(self.cell_matrix.shape[0]):
                if isinstance(self.cell_matrix[i][j], NoFuelState):
                    emoji_matrix[i][j] = '💧'
                elif isinstance(self.cell_matrix[i][j], FuelState):
                    emoji_matrix[i][j] = '🌳'
                elif isinstance(self.cell_matrix[i][j], BurningState):
                    emoji_matrix[i][j] = '🔥'
                elif isinstance(self.cell_matrix[i][j], BurntState):
                    emoji_matrix[i][j] = '🌑'
        print(emoji_matrix)

    def spark(self, row, col):
        self.cell_matrix[row][col] = BurningState(self.cell_matrix[row][col].probabilities)


def main():
    ones_matrix = np.ones((5, 5)) * 2
    half_matrix = np.ones((5, 5)) * 0.5
    print(half_matrix)
    cell_matrix = Matrix(ones_matrix, half_matrix, ones_matrix, ones_matrix, ones_matrix, ones_matrix)
    cell_matrix.render_emojis()
    cell_matrix.spark(0, 0)
    cell_matrix.render_emojis()
    for _ in range(0, 5):
        cell_matrix.update()
        cell_matrix.render_emojis()
        print("")


if __name__ == "__main__":
    main()