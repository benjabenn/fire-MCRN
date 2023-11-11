from __future__ import annotations
from enum import Enum
import numpy as np


class CellType(Enum):
    NO_FUEL = 1
    FUEL = 2
    BURNING = 3
    BURNT = 4


class CellProbabilities:
    """
    This class stores all of the probability values in a vector then involves numpy
    in the calculation of the p_burn value to optimize speed.
    """

    probability_vector: np.ndarray
    P_VECTOR_SIZE = (5, 1)

    def __init__(
        self, p_veg: float, p_den: float, p_wind: float, p_slope: float, p_h: float
    ):
        self.probability_vector = np.array(
            [p_veg, p_den, p_wind, p_slope, p_h])
        # Should this add 1 to p_veg and p_den before storing them in the vector
        # to have the product look like in the paper?

    @classmethod
    def from_vector(cls, probability_vector: np.ndarray):
        if probability_vector.shape != CellProbabilities.P_VECTOR_SIZE:
            raise Exception(
                f"Incorrect shape for vector parameter, try: {CellProbabilities.P_VECTOR_SIZE}"
            )
        return CellProbabilities(
            probability_vector[0],
            probability_vector[1],
            probability_vector[2],
            probability_vector[3],
            probability_vector[4],
        )

    def get_p_burn(self):
        return np.prod(self.probability_vector)


class Cell:
    probabilities: CellProbabilities

    def __init__(self, cell_probabilities: CellProbabilities):
        self.probabilities = cell_probabilities

    @classmethod
    def from_values(
        cls, p_veg: float, p_den: float, p_wind: float, p_slope: float, p_h: float
    ):
        return Cell(CellProbabilities(p_veg, p_den, p_wind, p_slope, p_h))

    def apply_rules(self, neighbors: list[Cell]):
        raise NotImplementedError("Subclasses must implement apply_rules.")

    def get_p_burn(self):
        return self.probabilities.get_p_burn()


class NoFuelState(Cell):
    def __init__(self, cell_probabilities: CellProbabilities):
        super().__init__(cell_probabilities)

    def apply_rules(self, neighbors: list[Cell]):
        return self


class FuelState(Cell):
    def __init__(self, cell_probabilities: CellProbabilities):
        super().__init__(cell_probabilities)

    def apply_rules(self, neighbors: list[Cell]):
        for neighbor in neighbors:
            if isinstance(neighbor, BurningState):
                if np.random.random() <= self.get_p_burn():
                    return BurningState(self.probabilities)
        return self


class BurningState(Cell):
    def __init__(self, cell_probabilities: CellProbabilities):
        super().__init__(cell_probabilities)

    def apply_rules(self, neighbors: list[Cell]):
        return BurntState(self.probabilities)


class BurntState(Cell):
    def __init__(self, cell_probabilities: CellProbabilities):
        super().__init__(cell_probabilities)

    def apply_rules(self, neighbors: list[Cell]):
        return self