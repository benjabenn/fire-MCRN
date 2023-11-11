from __future__ import annotations
from enum import Enum
import numpy as np


class CellType(Enum):
    """
    Simple enum class to represent CellStates.
    """
    NO_FUEL = 1
    FUEL = 2
    BURNING = 3
    BURNT = 4


class CellProbabilities:
    """
    This class stores all of the probability values in a vector then involves numpy
    in the calculation of the p_burn value to optimize speed.
    """
    probability_vector: np.ndarray  # Store our 5 values as a numpy vector to hopefully optimize calculation speed
    P_VECTOR_SIZE = (5, 1)  # Static size for our probability vector

    def __init__(
        self, p_veg: float, p_den: float, p_wind: float, p_slope: float, p_h: float
    ):
        """
        Constructor that takes in individual values as arguments.
        """
        self.probability_vector = np.array(
            [p_veg, p_den, p_wind, p_slope, p_h])
        # Should this add 1 to p_veg and p_den before storing them in the vector
        # to have the product look like in the paper?

    @classmethod
    def from_vector(cls, probability_vector: np.ndarray):
        """
        Alternate constructor that takes in the vector in a (5, 1) shape and returns 
        the CellProbabilities object made.
        """
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
        """
        Getting the p_burn value is as simple as multiplying all of the values in our 
        vector together with numpy's prod method.
        """
        return np.prod(self.probability_vector)


class Cell:
    probabilities: CellProbabilities

    def __init__(self, cell_probabilities: CellProbabilities):
        """
        Constructs a cell with a CellProbabilities object as an attribute.
        """
        self.probabilities = cell_probabilities

    @classmethod
    def from_values(
        cls, p_veg: float, p_den: float, p_wind: float, p_slope: float, p_h: float
    ):
        """
        Alternate constructor that takes in individual p values as arguments 
        instead of taking in a CellProbabilities object.
        """
        return Cell(CellProbabilities(p_veg, p_den, p_wind, p_slope, p_h))

    def apply_rules(self, neighbors: list[Cell]):
        """
        This method warns that Cell subclass objects must have their own implementation 
        of apply_rules.
        """
        raise NotImplementedError("Subclasses must implement apply_rules.")

    def get_p_burn(self):
        """
        Get the p_burn from the probabilities attribute of self.
        """
        return self.probabilities.get_p_burn()


class NoFuelState(Cell):
    def __init__(self, cell_probabilities: CellProbabilities):
        """
        Call the superclass (Cell) constructor with the CellProbabilities object
        """
        super().__init__(cell_probabilities)

    def apply_rules(self, neighbors: list[Cell]):
        """
        Apply rules for the NoFuelState: A NoFuelState will stay a NoFuelState because 
        it cannot catch on fire.
        """
        return self


class FuelState(Cell):
    def __init__(self, cell_probabilities: CellProbabilities):
        """
        Call the superclass (Cell) constructor with the CellProbabilities object
        """
        super().__init__(cell_probabilities)

    def apply_rules(self, neighbors: list[Cell]):
        """
        Apply rules for the FuelState: For every neighbor, if that neighbor is burning,
        run the p_burn simulation to see if the Cell will catch on fire. If it does, 
        the Cell becomes a BurningState object.
        """
        for neighbor in neighbors:
            if isinstance(neighbor, BurningState):
                if np.random.random() <= self.get_p_burn():
                    return BurningState(self.probabilities)
        return self


class BurningState(Cell):
    def __init__(self, cell_probabilities: CellProbabilities):
        """
        Call the superclass (Cell) constructor with the CellProbabilities object
        """
        super().__init__(cell_probabilities)

    def apply_rules(self, neighbors: list[Cell]):
        """
        Apply rules for the BurningState: BurningStates immediately become burnt on the 
        next time step.
        """
        return BurntState(self.probabilities)


class BurntState(Cell):
    def __init__(self, cell_probabilities: CellProbabilities):
        """
        Call the superclass (Cell) constructor with the CellProbabilities object
        """
        super().__init__(cell_probabilities)

    def apply_rules(self, neighbors: list[Cell]):
        """
        Apply rules for the BurntState: A BurntState will stay a BurntState because it 
        cannot catch on fire anymore.
        """
        return self
