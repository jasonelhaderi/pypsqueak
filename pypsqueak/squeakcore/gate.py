import copy
from typing import Tuple
import numpy as np
from pypsqueak.squeakcore._helpers import (
    _is_unitary,
    _is_power_of_two,
    _multi_arg_kronecker)


class Gate:
    '''
    A ``Gate`` is a variable-sized (shape is a tuple of powers of two), unitary
    matrix. Its state (returned by ``state()``) is a two-dimensional numpy
    array consisting of the computational basis representation of the quantum
    gate. By default it is initialized to the one qubit identity gate, but this
    can be overridden if the ``Gate`` is instantiated with some other
    numeric matrix as argument. If the matrix argument is not unitary, the
    ``Gate`` will fail to initialize. Additionally, the gate can be given a
    name via the corresponding kwarg. If not provided, defaults to ``None``.

    Note that ``len(some_gate)`` returns the number of qubits that
    ``some_gate`` acts on (``log2(some_gate.shape()[0])``)

    Examples
    --------

    >>> from pypsqueak.squeakcore import Gate
    >>> g1 = Gate()
    >>> g1
    [[1 0]
     [0 1]]
    >>> g1.state()
    array([[1, 0],
           [0, 1]])
    >>> g2 = Gate([(0, 1), (1, 0)])
    >>> g2
    [[0 1]
     [1 0]]
    >>> g3 = Gate(np.eye(4))
    >>> g3
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    >>> (len(g2), len(g3))
    (1, 2)
    >>> not_unitary = Gate([(0, 0), (1, 1)])
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    pypsqueak.errors.NonUnitaryInputError: Gate must be unitary.

    '''

    def __init__(self, some_matrix=[(1, 0), (0, 1)], name=None):

        self.__validate_gate(some_matrix, name)
        self.__state = np.array(some_matrix)
        self.__shape = (len(some_matrix), len(some_matrix[0]))

        if name is None:
            self.__name = str(self.__state)
        else:
            self.__name = name

    def state(self) -> np.ndarray:
        '''
        The state of the Gate as an ndarray.

        Returns
        -------
        numpy.ndarray
            A copy of the Gate's state.
        '''
        return np.copy(self.__state)

    def shape(self) -> Tuple[int, int]:
        '''
        Tuple of the Gate's shape. Equivalent to
        ``(2**len(some_gate),) * 2``.

        Returns
        -------
        tuple
            A copy of the Gate's shape.
        '''

        return copy.deepcopy(self.__shape)

    def name(self) -> str:
        '''
        Returns
        -------
        arbitrary
            The name of the ``Gate``.
        '''
        return self.__name

    def gate_product(self, *gates: 'Gate') -> 'Gate':
        '''
        Method for returning the Kronecker product of a gate with one or more
        other gates. When multiple arguments are specified, the product is
        computed sequentially from left to right.

        Note that this method does NOT have side-effects; it simply returns the
        product as a new Gate object.

        Returns the Kronecker product of a ``Gate`` with one or more other
        ``Gate`` instances.

        When multiple arguments are specified, the product is computed
        sequentially from the leftmost argument to the rightmost.

        Parameters
        ----------
        *gates : pypsqueak.squeakcore.Gate
            One or more ``Gate`` objects. Raises an exception if called with
            no arguments.

        Returns
        -------
        pypsqueak.squeakcore.Gate
            The left to right Kronecker product.

        Examples
        --------

        >>> from pypsqueak.squeakcore import Gate
        >>> g1 = Gate()
        >>> g2 = Gate([[0, 1], [1, 0]])
        >>> g1_g2 = g1.gate_product(g2)
        >>> g1_g2
        [[0 1 0 0]
         [1 0 0 0]
         [0 0 0 1]
         [0 0 1 0]]
        >>> g2_g1 = g2.gate_product(g1)
        >>> g2_g1
        [[0 0 1 0]
         [0 0 0 1]
         [1 0 0 0]
         [0 1 0 0]]

        '''

        num_args = len(gates)
        if num_args < 1:
            return Gate(self.__state)
        for argument in gates:
            if not isinstance(argument, type(Gate())):
                raise TypeError('Arguments must be Gate() objects.')

        product_gate = _multi_arg_kronecker(
            self.__state, *[gate.__state for gate in gates])
        return Gate(product_gate)

    @staticmethod
    def __validate_gate(potential_gate, gate_name):
        if _is_unitary(potential_gate):
            gate_shape = (len(potential_gate), len(potential_gate[0]))
        else:
            raise TypeError("Input matrix must be a numeric, "
                            "unitary nxn matrix (madeof nested lists, "
                            "tuples, or as an numpy array).")

        if not _is_power_of_two(gate_shape[0]) or gate_shape[0] == 1:
            raise TypeError('Gate must be nXn with n > 1 a power of 2.')

        if (not isinstance(gate_name, str)
                and not isinstance(gate_name, type(None))):
            raise TypeError('Name of Gate (if any) must be str.')

    def __len__(self):
        # Note that this returns the number of qubits the gate acts on, NOT the
        # size of matrix representation
        return int(np.log2(self.__shape[0]))

    def __repr__(self):
        if self.__name is not None:
            return self.__name

        return str(self.__state)
