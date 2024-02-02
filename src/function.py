import numpy as np
from numpy.typing import NDArray
from typing import Optional, List


class Function:
    key: str
    weights: List[int | float]
    is_present: bool
    index: Optional[int]
    input_id: Optional[str]
    num_indices: int
    missing: Optional[List[int]]

    def __init__(
        self,
        *,
        key: str,
        weights: np.array,
        is_present: bool,
        index: Optional[int] = None,
        input_id: Optional[str] = None,
        num_indices: int = -1,
        missing: Optional[List[int]] = None,
    ) -> None:
        self.key = key
        self.weights = weights
        self.is_present = is_present
        self.index = index
        self.input_id = input_id
        self.num_indices = num_indices
        self.missing = missing

    def compare_nonweights(self, function: "Function") -> bool:
        if self.key != function.key:
            return False
        if self.is_present != function.is_present:
            return False
        if self.index != function.index:
            return False
        if self.input_id != function.input_id:
            return False
        if self.num_indices != function.num_indices:
            return False
        if self.missing != function.missing:
            return False
        return True
