from typing import Tuple, List, Any

import numpy as np


# this is a class
# TODO: change class name
class Class():
    '''
    This is a class.

    Attributes
    ----------
    variable : int
        Description of variable (defaults to 0).
    '''

    def __init__(self) -> None:
        self.variable = 0

    def a_function(
        self,
        a: Tuple[int, int] = (0, 0),
        c: float = 0
    ) -> np.ndarray:
        '''
        This is a function.

        Parameters
        ----------
        a : Tuple[int, int], optional
            Description of a.

        Returns
        -------
        b : float
            Description of b.
        '''