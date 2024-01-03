from __future__ import annotations
import numpy as np
from cell import *
import time
import os


class Matrix:
    cell_matrix: np.ndarray
    time_steps: int
    time_steps_from_p_den: bool

    def __init__(
        self,
        cell_type_grid: np.ndarray,
        p_veg_grid: np.ndarray,
        p_den_grid: np.ndarray,
        p_wind_grid: np.ndarray,
        p_slope_grid: np.ndarray,
        p_h_grid: np.ndarray,
        time_steps: int = 1,
        time_step_from_p_den: bool = False,
    ):
        """
        Construct a Matrix object made up of Cell objects. The constructor takes in 6 equally shaped matrices:
            - cell_type_grid to determine what state each Cell begins in
            - p_veg_grid to represent p_veg values of the Cells based on real vegetation data
            - p_den_grid to represent p_den values of the Cells based on real vegetation density data
            - p_wind_grid to represent p_wind values of the Cells based on real windspeed data
            - p_slope_grid to represent p_slope values of the Cells based on real landscape slope data
            - p_h grid to represent the constant p_h value for each Cell
        Then for each matrix, at the (i, j) position a Cell is created in self.cell_matrix with the associated
        type or probability data from the corresponding matrix at the (i, j) position.
        """
        if not (
            cell_type_grid.shape
            == p_veg_grid.shape
            == p_den_grid.shape
            == p_wind_grid.shape
            == p_slope_grid.shape
            == p_h_grid.shape
        ):
            raise Exception("Grids must be the same shape!")
        self.time_steps = time_steps
        self.time_steps_from_p_den = time_step_from_p_den

        self.cell_matrix = np.empty(shape=cell_type_grid.shape, dtype=Cell)
        for i in range(self.cell_matrix.shape[1]):
            for j in range(self.cell_matrix.shape[0]):
                if cell_type_grid[i][j] == 1:
                    self.cell_matrix[i][j] = NoFuelState(
                        CellProbabilities(
                            p_veg_grid[i][j],
                            p_den_grid[i][j],
                            p_wind_grid[i][j],
                            p_slope_grid[i][j],
                            p_h_grid[i][j],
                        )
                    )
                elif cell_type_grid[i][j] == 2:
                    self.cell_matrix[i][j] = FuelState(
                        CellProbabilities(
                            p_veg_grid[i][j],
                            p_den_grid[i][j],
                            p_wind_grid[i][j],
                            p_slope_grid[i][j],
                            p_h_grid[i][j],
                        )
                    )
                elif cell_type_grid[i][j] == 3:
                    self.cell_matrix[i][j] = BurningState(
                        CellProbabilities(
                            p_veg_grid[i][j],
                            p_den_grid[i][j],
                            p_wind_grid[i][j],
                            p_slope_grid[i][j],
                            p_h_grid[i][j],
                        )
                    )
                elif cell_type_grid[i][j] == 4:
                    self.cell_matrix[i][j] = BurntState(
                        CellProbabilities(
                            p_veg_grid[i][j],
                            p_den_grid[i][j],
                            p_wind_grid[i][j],
                            p_slope_grid[i][j],
                            p_h_grid[i][j],
                        )
                    )

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
                new_cell_matrix[i][j] = self.cell_matrix[i][j].apply_rules(
                    self.get_neighbors(i, j),
                    self.time_steps
                    if not self.time_steps_from_p_den
                    else int(self.cell_matrix[i][j].probabilities.probability_vector[1] * 10),
                )
        self.cell_matrix = new_cell_matrix

    def render_emojis(self):
        emoji_matrix = np.empty_like(self.cell_matrix, dtype=object)
        for i in range(self.cell_matrix.shape[1]):
            for j in range(self.cell_matrix.shape[0]):
                if isinstance(self.cell_matrix[i][j], NoFuelState):
                    emoji_matrix[i][j] = "💧"
                elif isinstance(self.cell_matrix[i][j], FuelState):
                    emoji_matrix[i][j] = "🎄"
                elif isinstance(self.cell_matrix[i][j], BurningState):
                    emoji_matrix[i][j] = "🔥"
                elif isinstance(self.cell_matrix[i][j], BurntState):
                    emoji_matrix[i][j] = "🌑"
        print(emoji_matrix)

    def spark(self, row, col, allow_non_fuel_sparks=False):
        if (
            not isinstance(self.cell_matrix[row][col], FuelState)
            and not allow_non_fuel_sparks
        ):
            raise Exception(
                "Non-FuelState objects cannot be sparked without allow_non_fuel_sparks set to True."
            )
        else:
            self.cell_matrix[row][col] = BurningState(
                self.cell_matrix[row][col].probabilities
            )

    def is_complete(self):
        for i in range(self.cell_matrix.shape[1]):
            for j in range(self.cell_matrix.shape[0]):
                if isinstance(self.cell_matrix[i][j], BurningState):
                    return False
        return True

    def run_single_simulation(self, row, col) -> MatrixSimulationData:
        # self.render_emojis()
        self.spark(row, col)
        # self.render_emojis()
        burn_time = 0
        while not self.is_complete():
            self.update()
            # self.render_emojis()
            burn_time += 1
        return MatrixSimulationData(self.get_state_matrix(), burn_time)

    def get_state_matrix(self) -> np.ndarray:
        state_matrix = np.empty_like(self.cell_matrix)
        for i in range(self.cell_matrix.shape[1]):
            for j in range(self.cell_matrix.shape[0]):
                if isinstance(self.cell_matrix[i][j], NoFuelState):
                    state_matrix[i][j] = 1
                elif isinstance(self.cell_matrix[i][j], FuelState):
                    state_matrix[i][j] = 2
                elif isinstance(self.cell_matrix[i][j], BurningState):
                    state_matrix[i][j] = 3
                elif isinstance(self.cell_matrix[i][j], BurntState):
                    state_matrix[i][j] = 4
        return state_matrix

    def run_simulations(
        self, num_of_simulations, row=0, col=0, random_spark_location=False
    ) -> list[MatrixSimulationData]:
        final_states: list[MatrixSimulationData] = []
        

        for _ in range(num_of_simulations):
            starting_state = self.cell_matrix.copy()

            if random_spark_location:
                row = np.random.randint(0, self.cell_matrix.shape[1])
                col = np.random.randint(0, self.cell_matrix.shape[0])

            final_states.append(self.run_single_simulation(row, col))

            self.cell_matrix = starting_state
        return final_states
    
    def run_single_simulation_with_pauses(self, row, col) -> MatrixSimulationData:
        self.render_emojis()
        input()
        self.spark(row, col)
        self.render_emojis()
        input()
        burn_time = 0
        while not self.is_complete():
            self.update()
            self.render_emojis()
            input()
            burn_time += 1
        return MatrixSimulationData(self.get_state_matrix(), burn_time)


class MatrixSimulationData:
    final_state_matrix: np.ndarray
    burn_time: int

    def __init__(self, final_state_matrix: np.ndarray, burn_time: int) -> None:
        self.final_state_matrix = final_state_matrix
        self.burn_time = burn_time

    def __str__(self) -> str:
        return f"Matrix: \n{self.final_state_matrix}\nBurn Time: {self.burn_time}"
    
def get_average_final_matrix(input_data: list[MatrixSimulationData]) -> np.ndarray:
    if len(input_data) == 0:
        raise Exception("Error: No data found in list.")
    raise NotImplementedError()
    
def get_final_matrix_heatmap(input_data: list[MatrixSimulationData]) -> np.ndarray:
    if len(input_data) == 0:
        raise Exception("Error: No data found in list.")
    heatmap = np.zeros_like(input_data[0].final_state_matrix, dtype=float)
    for data_entry in input_data:
        for i in range(data_entry.final_state_matrix.shape[1]):
            for j in range(data_entry.final_state_matrix.shape[0]):
                if data_entry.final_state_matrix[i][j] == 4:
                    heatmap[i][j] += 1.0
    heatmap /= len(input_data)
    return heatmap


def main():
    ones_matrix = np.ones((10, 10))
    twos_matrix = np.ones((10, 10)) * 2
    half_matrix = np.ones((10, 10)) * 0.1
    cell_matrix = Matrix(
        twos_matrix, half_matrix, ones_matrix, ones_matrix, ones_matrix, ones_matrix, 5
    )

    start = time.time()
    data = cell_matrix.run_simulations(10000, 0, 0, True)
    print(get_final_matrix_heatmap(data))
    end = time.time()
    print(f"{end - start} seconds")
    # cell_matrix.run_single_simulation_with_pauses(0, 0)


if __name__ == "__main__":
    main()
