import numpy as np
# 1: Write method in Qubit which applies Gates, and scales properly with Tensor Product.
# 2: Create Measurement class for implementing measurements (start with 1 qubit, then scale up).
# 3: Build in standard gates and states.
# 4: Fine tune syntax.
# 5: Test out 5-Qubit Deutsch-Jozsa Algorithm.



class Qubit:
    '''Creates a normalized Qubit object out of some_vector.'''

    def __init__(self, init_state = [1, 0]):
        # Checks that input is valid.
        self.validate_state(init_state)

        # Initialize qubit.
        self.state = np.array(init_state)
        self.normalize()

    def validate_state(self, some_vector):
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
            raise NullVectorError('state cannot be the null vector.')

        # Checks that some_vector has length 2n with integer n.
        elif len(some_vector) % 2 != 0:
            raise WrongShapeError('Input state must have even length.')

    def change_state(self, new_state):
        # Checks that input is valid.
        self.validate_state(new_state)

        # Changes the state.
        self.state = new_state
        self.normalize()

    def normalize(self):
        dual_state = np.conjugate(self.state)
        norm = np.sqrt(np.dot(self.state, dual_state))
        self.state = np.multiply(1/norm, self.state)

    def print_state(self):
        # Prints the state in the computational basis.
        padding = len(format(len(self.state), 'b')) - 1
        label = format(0, 'b').zfill(padding)
        amplitude = self.state[0]
        print("({0:.2e})|{1}>".format(amplitude, label), end='')

        for i in range(1, len(self.state)):
            label = format(i, 'b').zfill(padding)
            amplitude = self.state[i]
            print(" + ({:.2e})|{}>".format(amplitude, label), end='')

        print('')

    def qubit_product(self, *arg):
        if len(arg) == 0:
            raise TypeError('Input cannot be empty.')
        new_qubits = self.state

        for argument in arg:
            if not isinstance(argument, type(Qubit())):
                raise TypeError('Arguments must be Qubit() objects.')

        if len(arg) == 1:
            new_qubits = np.kron(new_qubits, arg[0].state)
            return Qubit(new_qubits.tolist())

        if len(arg) > 1:
            for argument in arg:
                new_qubits = np.kron(new_qubits, argument.state)
            return Qubit(new_qubits.tolist())
        pass


class Gate:

    def __init__(self, some_matrix = [(1, 0), (0, 1)]):
        # Checks that input is list-like
        if not isinstance(some_matrix, list) and not isinstance(some_matrix, tuple):
            raise ValueError('Input must be list or tuple.')

        # Checks that input is matrix-like
        elif any(not isinstance(element, list) and not isinstance(element, tuple)\
                                                    for element in some_matrix):
            raise ValueError('Elements of input must be list or tuple.')

        # Checks that the input is a square matrix
        self.shape = (len(some_matrix), len(some_matrix[0]))
        if self.shape[0] % 2 != 0:
            raise WrongShapeError('Gate must be nXn with even n.')

        for row in some_matrix:
            if len(row) != self.shape[0]:
                raise WrongShapeError('Gate must be a square matrix.')

        # Initialize the gate
        self.state = np.array(some_matrix)

        # Checks that the input is unitary
        product_with_conj = np.dot(self.state.conj().T, self.state)
        is_unitary = np.allclose(product_with_conj, np.eye(self.shape[0]))
        if is_unitary == False:
            raise NonUnitaryInputError('Gate must be unitary.')

    def gate_product(self, *arg):
        # Returns the a Gate() that is the Kronecker product of self and *args
        new_gate = self.state
        if len(arg) == 0:
            raise TypeError('Input must not be empty.')

        for argument in arg:
            if not isinstance(argument, type(Gate())):
                raise TypeError('Arguments must be Gate() objects.')

        if len(arg) == 1:
            new_gate = np.kron(new_gate, arg[0].state)
            return Gate(new_gate.tolist())

        if len(arg) > 1:
            for argument in arg:
                new_gate = np.kron(new_gate, argument.state)
            return Gate(new_gate.tolist())

    def print_state(self):
        print(self.state)


# Custom errors.
class WrongShapeError(ValueError):
    pass

class NullVectorError(ValueError):
    pass

class InhomogenousInputError(TypeError):
    pass

class NonUnitaryInputError(ValueError):
    pass

# Tests

# q0 = Qubit([1, 0])
# q1 = Qubit([0, 1])
# q = Qubit((10j + 4, 17j - 1, 3, 0))
# q0.print_state()
# q1.print_state()
# q.qubit_product(q).print_state()
# g = Gate()
# g.print_state()
# print(g.state)
# print(q1.state)
# two_qubits = TensorProduct(q0, q1)
# three_qubits = TensorProduct(q0, q1, q)
# double_identity = TensorProduct(g, g)
# print(two_qubits.state)
# print(three_qubits.state)
# two_qubits.print_state()
# three_qubits.print_state()
# print(double_identity.state)
