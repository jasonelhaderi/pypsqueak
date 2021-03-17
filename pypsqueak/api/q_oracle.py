import numpy as np
from .q_op import qOp
from ._helpers import _validateClassicalFunction


class qOracle(qOp):
    '''
    Subclass of ``qOp``. Useful for representing quantum black-boxes, such as
    that appearing in the Deutsch-Jozsa algorithm.

    ``qOracle`` implements a unitary transformation U_f|x>|y> = |x>|y XOR f(x)>
    where the classical function f maps nonnegative integers to nonnegative
    integers. The XOR is performed bitwise on the computational basis label y
    and f(x). This reduces to mod 2 addition when y and f(x) are both one bit
    long. ``n`` specifies the number of qubits in the left side portion of the
    quantum register, while ``m`` specifies the number of qubits in the right
    side portion.
    '''

    def __init__(self, func, n, m=1, kraus_ops=None):
        if (not isinstance(n, int) or not isinstance(m, int)
                or n < 1 or m < 1):
            raise TypeError(
                'Dimension exponents n and m must be positive integers.')

        _validateClassicalFunction(func, n, m)

        self.__classical_func = func
        self.__domain_exp = n
        self.__range_exp = m

        super().__init__(self.__generate_matrix_rep(), kraus_ops=kraus_ops)

    def classical_func(self, x_val):
        '''
        Returns the classical value of the function implemented by the
        ``qOracle``. Raises an exception if the argument isn't nonnegative or
        if larger than the ``n`` portion of the intended register.

        Parameters
        ----------
        x_val : int
            Argument of function.

        Returns
        -------
        int
            Value of the function.

        Examples
        --------

        >>> from pypsqueak.api import qOracle
        >>> black_box = qOracle(lambda x: 0, 2)
        >>> black_box.classical_func(1)
        0
        >> black_box.classical_func(3)
        0
        >> black_box.classical_func(3)
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "pypsqueak/api.py", line 1065, in classical_func
            raise ValueError("Classical function input out of bounds.")
        ValueError: Classical function input out of bounds.

        '''

        if not isinstance(x_val, int):
            raise TypeError("Classical function maps ints to ints.")
        if x_val < 0 or x_val > 2**self.__domain_exp - 1:
            raise ValueError("Classical function input out of bounds.")

        return self.__classical_func(x_val)

    def __generate_matrix_rep(self):
        '''
        Generates the computational basis matrix representation of the oracle
        for the register in state |x>|y>.
        '''

        numberOfQubits = self.__range_exp + self.__domain_exp
        matrixRepOfOracle = np.zeros((2**numberOfQubits, 2**numberOfQubits))
        for x in range(2**self.__domain_exp):
            for y in range(2**self.__range_exp):
                row = int(
                    "{0:b}{1:b}".format(x, y ^ self.classical_func(x)), 2)
                col = int("{0:b}{1:b}".format(x, y), 2)
                matrixRepOfOracle[row][col] = 1

        return matrixRepOfOracle

    def __repr__(self):
        return "qOracle({}, {})".format(self.__domain_exp, self.__range_exp)
