'''
Provides core ``Qubit`` and ``Gate`` objects that function as the simulated
backend of pypSQUEAK.
'''

import numpy as np
import copy
import pypsqueak.errors as sqerr

class Qubit:
    '''
    A ``Qubit`` is a variable-sized (length can be powers of two), normalized,
    complex vector. Its state (returned by ``state()``) is a one-dimensional
    numpy array consisting of the computational basis representation of the quantum
    state. By default it is initialized in the |0> state, but this can be
    overridden if a ``Qubit`` is instantiated with some other numeric vector
    as argument (the resulting ``Qubit`` will use the normalized version of
    that vector).

    The state a ``Qubit`` can be changed with a call to ``Qubit.change_state()``.
    Additionally, a dictionary with computational basis state labels as keys
    and corresponding components as values gets returned by
    ``Qubit.computational_decomp()``.

    Note that the length of a ``Qubit`` is the number of qubits that the state
    vector corresponds to (``log2(len(state_vector))``).

    Examples
    --------
    Here we initialize a ``Qubit`` in the |0> state, and then change it to the
    |11> state.

    >>> from pypsqueak.squeakcore import Qubit
    >>> q = Qubit()
    >>> p = Qubit([0, 1, 0, 0])
    >>> q
    [1. 0.]
    >>> print(q)
    (1.00e+00)|0> + (0.00e+00)|0>
    >>> print(p)
    (0.00e+00)|00> + (1.00e+00)|01> + (0.00e+00)|10> +(0.00e+00)|11>
    >>> q.change_state([0, 0, 0, 1])
    >>> print(q)
    (0.00e+00)|00> + (0.00e+00)|01> + (0.00e+00)|10> + (1.00e+00)|11>

    '''

    def __init__(self, init_state = [1, 0]):
        # Checks that input is valid.
        self.__validate_state(init_state)

        # Initialize qubit.
        self.__state = np.array(init_state)
        self.__normalize()
        self.__decompose_into_comp_basis()

    def __validate_state(self, some_vector):
        # Checks that some_vector is a list or tuple.
        if type(some_vector) != list and type(some_vector) != tuple:
            if not isinstance(some_vector, type(np.array([0]))):
                raise TypeError('Input state must be a list, tuple, or numpy array.')

        # Checks that elements of some_vector are numeric.
        for element in some_vector:
            try:
                element + 5
            except:
                raise TypeError('Elements of input state must be numeric.')

        # Checks that the some_vector isn't null, or the null vector.
        if all(element == 0 for element in some_vector):
            raise sqerr.NullVectorError('State cannot be the null vector.')

        # Checks that some_vector has length greater than 1 which is a power of 2.
        if not sqerr._is_power_2(len(some_vector)) or len(some_vector) == 1:
            raise sqerr.WrongShapeError('Input state must have a length > 1 which is a power of 2.')

    def change_state(self, new_state):
        '''
        Changes the state of the Qubit to the normalized vector corresponding to
        the argument.

        Parameters
        ----------
        new_state : list-like
            The vector representation of the new state in the computational
            basis. Must have a length which is a power of two.

        Returns
        -------
        None
            The Qubit instance on which ``change_state()`` is called is
            altered.
        '''
        # Checks that input is valid.
        self.__validate_state(new_state)

        # Changes the state.
        self.__state = np.array(new_state)
        self.__normalize()
        self.__decompose_into_comp_basis()

    def state(self):
        '''
        The state of the Qubit as an ndarray.

        Returns
        -------
        numpy.ndarray
            A copy of the Qubit's state.
        '''

        return np.copy(self.__state)

    def computational_decomp(self):
        '''
        A representation of the Qubit's state via a dict. Computational basis
        labels are the keys and the components of the Qubit are the values.

        Returns
        -------
        dict
            A computational basis representation of the Qubit.
        '''

        return copy.deepcopy(self.__computational_decomp)

    def __normalize(self):
        dual_state = np.conjugate(self.__state)
        norm = np.sqrt(np.dot(self.__state, dual_state))
        self.__state = np.multiply(1/norm, self.__state)
        self.__decompose_into_comp_basis()

    def __decompose_into_comp_basis(self):
        # Generates a dict with basis state labels as keys and amplitudes as values
        self.__computational_decomp = {}
        padding = len(format(len(self.__state), 'b')) - 1
        label = format(0, 'b').zfill(padding)
        amplitude = self.__state[0]
        self.__computational_decomp[label] = amplitude

        for i in range(1, len(self.__state)):
            label = format(i, 'b').zfill(padding)
            amplitude = self.__state[i]
            self.__computational_decomp[label] = amplitude

    def __len__(self):
        # Note that this returns the number of qubits that the given Qubit object
        # corresponds to, not the number of components its vector representation has
        return int(np.log2(len(self.__state)))

    def __repr__(self):
        return str(self.__state)

    def __str__(self):
        # Generates a string representation of the state in the computational basis
        first_term_flag = 0
        state_rep = ""
        for state_label in self.__computational_decomp:
            # This if statement supresses terms with zero magnitude.
            if not self.__computational_decomp[state_label] == 0\
                and not self.__computational_decomp[state_label] == 0 + 0j:
                if first_term_flag == 0:
                    state_rep += "({0:.2e})|{1}>".format(self.__computational_decomp[state_label], state_label)
                    first_term_flag = 1

                elif first_term_flag == 1:
                    state_rep += " + ({:.2e})|{}>".format(self.__computational_decomp[state_label], state_label)
            else:
                pass

        return state_rep

    def qubit_product(self, *arg):
        '''
        Returns the Kronecker product of a ``Qubit`` with one or more other
        ``Qubit`` objects.

        When multiple arguments are specified, the product is computed
        sequentially from the leftmost argument to the rightmost.

        Parameters
        ----------
        *arg : pypsqueak.squeakcore.Qubit
            One or more ``Qubit`` objects. Raises an exception if called with
            no arguments.

        Returns
        -------
        pypsqueak.squeakcore.Qubit
            The left to right Kronecker product.

        Examples
        --------

        >>> from pypsqueak.squeakcore import Qubit
        >>> q1 = Qubit()
        >>> q2 = Qubit([0, 1])
        >>> q3 = Qubit([1, 0, 0, 0])
        >>> q1_q2 = q1.qubit_product(q2)
        >>> q1_q2
        [1. 0.]
        >>> q1_q2.state()
        array([0., 1., 0., 0.])
        >>> q2_q1 = q2.qubit_product(q1)
        >>> q2_q1
        [0. 0. 1. 0.]
        >>> q1_q2_q3 = q1.qubit_product(q2, q3)
        >>> q1_q2_q3
        [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

        '''
        if len(arg) == 0:
            raise TypeError('Must specify at least one argument.')
        new_qubits = self.__state

        for argument in arg:
            if not isinstance(argument, type(Qubit())):
                raise TypeError('Arguments must be Qubit() objects.')

        if len(arg) == 1:
            new_qubits = np.kron(new_qubits, arg[0].state())
            return Qubit(new_qubits)

        if len(arg) > 1:
            for argument in arg:
                new_qubits = np.kron(new_qubits, argument.state())
            return Qubit(new_qubits)

class Gate:
    '''
    A ``Gate`` is a variable-sized (shape is a tuple of powers of two), unitary
    matrix. Its state (returned by ``state()``) is a two-dimensional numpy array
    consisting of the computational basis representation of the quantum
    gate. By default it is initialized to the one qubit identity gate, but this
    can be overridded if the ``Gate`` is instantiated with some other numeric matrix
    as argument. If the matrix argument is not unitary, the ``Gate`` will fail
    to initialize. Additionally, the gate can be given a name via the
    the corresponding kwarg. If not provided, defaults to ``None``.

    Note that ``len(some_gate)`` returns the number of qubits that ``some_gate``
    acts on (``log2(some_gate.shape()[0])``)

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

    def __init__(self, some_matrix = [(1, 0), (0, 1)], name = None):
        # Checks that input is list-like
        if not isinstance(some_matrix, list) and not isinstance(some_matrix, tuple):
            if not isinstance(some_matrix, type(np.array([0]))):
                raise TypeError('Input must be list, tuple, or numpy array.')

        # Checks that input is matrix-like
        elif any(not isinstance(element, list) and not isinstance(element, tuple)\
                                                    for element in some_matrix):
            raise TypeError('Input must be list-like of list-likes.')

        # Checks that the elements of input are numeric.
        for row in some_matrix:
            for element in row:
                try:
                    element + 5
                except:
                    raise TypeError("Elements of input must be numeric.")

        # Checks that the input is a square matrix
        self.__shape = (len(some_matrix), len(some_matrix[0]))
        if not sqerr._is_power_2(self.__shape[0]) or self.__shape[0] == 1:
            raise sqerr.WrongShapeError('Gate must be nXn with n > 1 a power of 2.')

        for row in some_matrix:
            if len(row) != self.__shape[0]:
                raise sqerr.WrongShapeError('Input not a square matrix.')

        # Checks that the name (if any) is a string
        if not isinstance(name, str) and not isinstance(name, type(None)):
            raise TypeError('Name of Gate (if any) must be str.')

        # Initialize the gate
        self.__state = np.array(some_matrix)
        if name == None:
            self.__name = str(self.__state)

        if name != None:
            self.__name = name

        # Checks that the input is unitary
        product_with_conj = np.dot(self.__state.conj().T, self.__state)
        is_unitary = np.allclose(product_with_conj, np.eye(self.__shape[0]))
        if is_unitary == False:
            raise sqerr.NonUnitaryInputError('Gate must be unitary.')

    def state(self):
        '''
        The state of the Gate as an ndarray.

        Returns
        -------
        numpy.ndarray
            A copy of the Gate's state.
        '''
        return np.copy(self.__state)

    def shape(self):
        '''
        Tuple of the Gate's shape. Equivalent to
        ``(2**len(some_gate),) * 2``.

        Returns
        -------
        tuple
            A copy of the Gate's shape.
        '''

        return copy.deepcopy(self.__shape)

    def name(self):
        '''
        Returns
        -------
        arbitrary
            The name of the ``Gate``.
        '''
        return self.__name

    def gate_product(self, *arg):
        '''
        Method for returning the Kronecker product of a gate with one or more
        other gates. When multiple arguments are specified, the product is
        computed sequentially from left to right.

        Note that this method does NOT have side-effects; it simply returns the
        product as a new Gate object.

        Returns the Kronecker product of a ``Gate`` with one or more other ``Gate``s.

        When multiple arguments are specified, the product is computed
        sequentially from the leftmost argument to the rightmost.

        Parameters
        ----------
        \*arg : pypsqueak.squeakcore.Gate
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

        new_gate = self.__state
        if len(arg) == 0:
            return Gate(new_gate)

        for argument in arg:
            if not isinstance(argument, type(Gate())):
                raise TypeError('Arguments must be Gate() objects.')

        if len(arg) == 1:
            new_gate = np.kron(new_gate, arg[0].state())
            return Gate(new_gate)

        if len(arg) > 1:
            for argument in arg:
                new_gate = np.kron(new_gate, argument.state())
            return Gate(new_gate)

    def __len__(self):
        # Note that this returns the number of qubits the gate acts on, NOT the
        # size of matrix representation
        return int(np.log2(self.__shape[0]))

    def __repr__(self):
        if self.__name != None:
            return self.__name

        else:
            return str(self.__state)
