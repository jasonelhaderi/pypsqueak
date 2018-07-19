import numpy as np
import copy
import squal.errors as sqerr
# import squal.gates as gt

# 1: Fine tune syntax by writing a front-end API to interact with the simulation.
#    (a) Change the Gate class so that it has the attribute target_qubit, so that
#    the functions in gates.py return Gate objects. Adapt QCSim and unit tests
#    accordingly. (DONT DO, MAKE API HANDLE)
#    (b) Flesh out Program class in API with means of doing measurement. (DONE)
#    (c) Fix unit tests. (DONE)
#    (d) Fix imports in files. (DONE)
# 2: Build in standard gates as functions. Write tests for multi-qubit gates.
#    Add all necessary control operations for classical bits and other functions.
# 3: Test out 5-Qubit Deutsch-Jozsa Algorithm.
# 4: Implement noise modeling.
# 4: Generate QASM and Quil descriptions of code.
'''
Core components of squal are defined here.
'''

class Qubit:
    '''
    Creates a normalized Qubit object out of some_vector.
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
            raise TypeError('Input state must be a list or tuple.')

        # Checks that elements of some_vector are numeric.
        elif any(isinstance(element, list) for element in some_vector):
            raise TypeError('Elements of input state cannot be lists.')

        elif any(isinstance(element, tuple) for element in some_vector):
            raise TypeError('Elements of input state cannot be tuples.')

        elif any(isinstance(element, dict) for element in some_vector):
            raise TypeError('Elements of input state cannot be dicts.')

        elif any(isinstance(element, str) for element in some_vector):
            raise TypeError('Elements of input state cannot be strings.')

        # Checks that the some_vector isn't null, or the null vector.
        elif all(element == 0 for element in some_vector):
            raise sqerr.NullVectorError('State cannot be the null vector.')

        # Checks that some_vector has length 2n with integer n.
        elif len(some_vector) % 2 != 0:
            raise sqerr.WrongShapeError('Input state must have even length.')

    def change_state(self, new_state):
        # Checks that input is valid.
        self.__validate_state(new_state)

        # Changes the state.
        self.__state = np.array(new_state)
        self.__normalize()
        self.__decompose_into_comp_basis()

    def state(self):
        '''
        Returns a copy of self.__state for use in operations.
        '''

        return np.copy(self.__state)

    def computational_decomp(self):
        '''
        Returns a copy of self.__computational_decomp for use in operations.
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
            if first_term_flag == 0:
                state_rep += "({0:.2e})|{1}>".format(self.__computational_decomp[state_label], state_label)
                first_term_flag = 1

            elif first_term_flag == 1:
                state_rep += " + ({:.2e})|{}>".format(self.__computational_decomp[state_label], state_label)

        return state_rep

    def qubit_product(self, *arg):
        if len(arg) == 0:
            raise TypeError('Input cannot be empty.')
        new_qubits = self.__state

        for argument in arg:
            if not isinstance(argument, type(Qubit())):
                raise TypeError('Arguments must be Qubit() objects.')

        if len(arg) == 1:
            new_qubits = np.kron(new_qubits, arg[0].state())
            return Qubit(new_qubits.tolist())

        if len(arg) > 1:
            for argument in arg:
                new_qubits = np.kron(new_qubits, argument.state())
            return Qubit(new_qubits.tolist())

class Gate:
    '''
    Creates a unitary gate out of some_matrix.
    '''

    def __init__(self, some_matrix = [(1, 0), (0, 1)], name = None):
        # Checks that input is list-like
        if not isinstance(some_matrix, list) and not isinstance(some_matrix, tuple):
            raise ValueError('Input must be list or tuple.')

        # Checks that input is matrix-like
        elif any(not isinstance(element, list) and not isinstance(element, tuple)\
                                                    for element in some_matrix):
            raise ValueError('Elements of input must be list or tuple.')

        # Checks that the input is a square matrix
        self.__shape = (len(some_matrix), len(some_matrix[0]))
        if self.__shape[0] % 2 != 0:
            raise sqerr.WrongShapeError('Gate must be nXn with even n.')

        for row in some_matrix:
            if len(row) != self.__shape[0]:
                raise sqerr.WrongShapeError('Gate must be a square matrix.')

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
        Returns a copy of self.__state for use in operations.
        '''
        return np.copy(self.__state)

    def shape(self):
        return copy.deepcopy(self.__shape)

    def gate_product(self, *arg):
        # Returns the a Gate() that is the Kronecker product of self and *args
        new_gate = self.__state
        if len(arg) == 0:
            return Gate(new_gate.tolist())

        for argument in arg:
            if not isinstance(argument, type(Gate())):
                raise TypeError('Arguments must be Gate() objects.')

        if len(arg) == 1:
            new_gate = np.kron(new_gate, arg[0].state())
            return Gate(new_gate.tolist())

        if len(arg) > 1:
            for argument in arg:
                new_gate = np.kron(new_gate, argument.state())
            return Gate(new_gate.tolist())

    def __len__(self):
        # Note that this returns the number of qubits the gate acts on, NOT the
        # size of matrix representation
        return int(np.log2(self.__shape[0]))

    def __repr__(self):
        if self.__name != None:
            return self.__name

        else:
            return str(self.__state)
