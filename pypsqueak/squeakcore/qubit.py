import numpy as np
import copy
from pypsqueak.squeakcore._helpers import (_is_power_of_two, _is_listlike,
                                           _has_only_numeric_elements,
                                           _is_normalizable)
from pypsqueak.errors import NullVectorError, WrongShapeError


class Qubit:
    '''
    A ``Qubit`` is a variable-sized (length can be powers of two), normalized,
    complex vector. Its state (returned by ``state()``) is a one-dimensional
    numpy array consisting of the computational basis representation of the
    quantum state. By default it is initialized in the |0> state, but this can
    be overridden if a ``Qubit`` is instantiated with some other numeric vector
    as argument (the resulting ``Qubit`` will use the normalized version of
    that vector).

    The state a ``Qubit`` can be changed with a call to
    ``Qubit.change_state()``. Additionally, a dictionary with computational
    basis state labels as keys and corresponding components as values gets
    returned by ``Qubit.computational_decomp()``.

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

    def __init__(self, init_state=[1, 0]):

        self.__state = None
        self.__computational_decomp = None
        self.change_state(init_state)

    def change_state(self, new_state):
        '''
        Changes the state of the Qubit to the normalized vector corresponding
        to the argument.

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
        Note that the basis state labels (i.e. the keys) are big-endian.

        Returns
        -------
        dict
            A computational basis representation of the Qubit.
        '''

        return copy.deepcopy(self.__computational_decomp)

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

        for argument in arg:
            if not isinstance(argument, type(Qubit())):
                raise TypeError('Arguments must be Qubit() objects.')

        product_state = self.__state

        for argument in arg:
            product_state = np.kron(product_state, argument.state())

        return Qubit(product_state)

    def __validate_state(self, some_vector):
        if not _is_listlike(some_vector):
            raise TypeError("Input state must be a list, "
                            "tuple, or numpy array.")

        if not _has_only_numeric_elements(some_vector):
            raise TypeError('Elements of input state must be numeric.')

        if not _is_normalizable(some_vector):
            raise NullVectorError('State cannot be the null vector.')

        if not _is_power_of_two(len(some_vector)) or len(some_vector) == 1:
            raise WrongShapeError("Input state must have a length > 1 "
                                  "which is a power of 2.")

    def __normalize(self):

        dual_state = np.conjugate(self.__state)
        norm = np.sqrt(np.dot(self.__state, dual_state))
        self.__state = np.multiply(1/norm, self.__state)
        self.__decompose_into_comp_basis()

    def __decompose_into_comp_basis(self):
        '''
        Generates a dict with basis state labels as keys and amplitudes as
        values. The result is stored internally.
        '''

        self.__computational_decomp = {}
        number_of_qubits = len(self)

        # Loop over self.__state since we need each basis state amplitude.
        for basis_state in range(0, len(self.__state)):
            basis_label = self.__convert_to_binary_string(
                basis_state, number_of_qubits)
            amplitude = self.__state[basis_state]
            self.__computational_decomp[basis_label] = amplitude

    def __convert_to_binary_string(self, number, length):
        '''
        Returns ``number`` as a binary string with length ``length``,
        filling in leading zeros if necessary.
        '''

        return format(number, 'b').zfill(length)

    def __len__(self):
        '''
        The number of qubits that the given Qubit corresponds to,
        not the number of components its vector representation has.
        '''

        return int(np.log2(len(self.__state)))

    def __repr__(self):

        return "Qubit({})".format(repr(self.__state))

    def __str__(self):
        '''
        Generates a string representation of the state in the computational
        basis.
        '''

        string_representation = ""

        for basis_state_label in self.__computational_decomp:
            if self.__computational_decomp[basis_state_label] == 0 + 0j:
                continue
            elif len(string_representation) == 0:
                string_representation += self.__make_basis_term_string_rep(
                    self.__computational_decomp[basis_state_label],
                    basis_state_label
                )
            else:
                string_representation += " + " +\
                    self.__make_basis_term_string_rep(
                        self.__computational_decomp[basis_state_label],
                        basis_state_label
                    )

        return string_representation

    def __make_basis_term_string_rep(self, amplitude, basis_state_label):

        if not isinstance(amplitude, complex):
            term_rep = "({:.2e})|{}>".format(amplitude, basis_state_label)
        elif amplitude == 0 + 0j:
            term_rep = "(0)|{}>".format(basis_state_label)
        elif amplitude.imag == 0:
            term_rep = "({:.2e})|{}>".format(amplitude.real, basis_state_label)
        elif amplitude.real == 0:
            term_rep = "({:.2e}j)|{}>".format(
                amplitude.imag, basis_state_label)
        else:
            term_rep = "({:.2e})|{}>".format(amplitude, basis_state_label)

        return term_rep
