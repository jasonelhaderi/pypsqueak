import numpy as np

# Add support for printing qubits, gates, and tensor products nicely
# Write method in Qubit which applies Gates, and scales properly with Tensor Product
# Create Measurement class for implementing measurements (start with 1 qubit, then scale up)
# Build in standard gates and states
# Fine tune syntax
# Test out 5-Qubit Deutsch-Jozsa Algorithm

class Qubit:
    '''Creates a normalized Qubit object out of some_vector.'''

    def __init__(self, init_state = [1, 0]):
        # Checks that input is valid
        self.validate_state(init_state)

        # Initialize qubit
        self.state = np.array(init_state)
        self.normalize()

    def validate_state(self, some_vector):
        # Checks that some_vector is a list or tuple
        if type(some_vector) != list and type(some_vector) != tuple:
            raise TypeError('Input state must be a list or tuple')

        # Checks that elements of some_vector are numeric
        elif any(isinstance(element, list) for element in some_vector):
            raise TypeError('Elements of input state cannot be lists')

        elif any(isinstance(element, tuple) for element in some_vector):
            raise TypeError('Elements of input state cannot be tuples')

        elif any(isinstance(element, dict) for element in some_vector):
            raise TypeError('Elements of input state cannot be dicts')

        elif any(isinstance(element, str) for element in some_vector):
            raise TypeError('Elements of input state cannot be strings')

        # Checks that the some_vector isn't null, or the null vector
        elif all(element == 0 for element in some_vector):
            raise NullVectorError('state cannot be the null vector')

        # Checks that some_vector has length 2
        elif len(some_vector) != 2:
            raise WrongShapeError('Input state must have length 2')

    def change_state(self, new_state):
        # Checks that input is valid
        self.validate_state(new_state)

        # Changes the state
        self.state = new_state
        self.normalize()

    def normalize(self):
        dual_state = np.conjugate(self.state)
        norm = np.sqrt(np.dot(self.state, dual_state))
        self.state = np.multiply(1/norm, self.state)

class Gate:

    def __init__(self, some_matrix = [(1, 0), (0, 1)]):
        # Checks that input is list-like
        if not isinstance(some_matrix, list) and not isinstance(some_matrix, tuple):
            raise ValueError('Input must be list or tuple')

        # Checks that input is matrix-like
        elif any(not isinstance(element, list) and not isinstance(element, tuple)\
                                                    for element in some_matrix):
            raise ValueError('Elements of input must be list or tuple')

        # Checks that the input is a sensible 2X2 matrix
        elif len(some_matrix) != 2:
            raise WrongShapeError('Gate must have shape (2, 2)')

        elif len(some_matrix[0]) != 2:
            raise WrongShapeError('Gate must have shape (2, 2)')

        elif len(some_matrix[0]) != len(some_matrix[-1]):
            raise WrongShapeError('Gate must have shape (2, 2)')

        # Initialize the gate
        self.state = np.array(some_matrix)


class TensorProduct:
    '''Implements a tensor product as a list of all qubits or all operators.'''

    def __init__(self, *arg):
        self.parts = []

        # Check that arg isn't empty
        if len(arg) == 0:
            raise TypeError('Input must be nonempty')

        if len(arg) != 0:

            # Check that all the args are either Qubits() or Gates()
            if any(not isinstance(argument, type(Qubit())) and not isinstance(argument, type(Gate())) for argument in arg):
                raise TypeError('All arguments must have type Qubit() or Gate()')

            # Check that argument type is homogenous
            elif not all(isinstance(argument, type(arg[0])) for argument in arg):
                raise InhomogenousInputError('TensorProduct() can only be take on arguments of same type')

            # Place qubit or gate objects into list
            for item in arg:
                self.parts.append(item)


# Custom Errors
class WrongShapeError(ValueError):
    pass

class NullVectorError(ValueError):
    pass

class InhomogenousInputError(TypeError):
    pass

q0 = Qubit([1, 0])
q1 = Qubit([0, 1])
g = Gate()
# print(g.state)
# print(q1.state)
two_qubits = TensorProduct(q0, q1)
# print(two_qubits.parts[1].state)
# print(two_qubits.state)

# print([qubit.state for qubit in two_qubits.parts])
