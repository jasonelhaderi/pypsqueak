import numpy as np

# 1: Add support for printing qubits(DONE), gates(DONE), and tensor products nicely(DONE).
# 2: Add check to gates that unitary(DONE).
# 3: Add check to TensorProduct that makes sure unitary if gate, normalized if qubits.
# 4: Accomplish above line by making ProductOfGates and ProductOfQubits to
#    replace TensorProduct and have them inherit from Gate and Qubit. This would
#    entail eith modifying Qubit and Gate to not be length 2 and 2x2, or undoing
#    this requirement in the child Product-classes.
# 5: Write method in Qubit which applies Gates, and scales properly with Tensor Product.
# 6: Create Measurement class for implementing measurements (start with 1 qubit, then scale up).
# 7: Build in standard gates and states.
# 8: Fine tune syntax.
# 9: Test out 5-Qubit Deutsch-Jozsa Algorithm.
# optional: Remove self.parts from Tensor product?


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

        # Checks that some_vector has length 2.
        elif len(some_vector) != 2:
            raise WrongShapeError('Input state must have length 2.')

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
        print("({:.2e})|0> + ({:.2e})|1>".format(self.state[0], self.state[1]))

class Gate:

    def __init__(self, some_matrix = [(1, 0), (0, 1)]):
        # Checks that input is list-like.
        if not isinstance(some_matrix, list) and not isinstance(some_matrix, tuple):
            raise ValueError('Input must be list or tuple.')

        # Checks that input is matrix-like.
        elif any(not isinstance(element, list) and not isinstance(element, tuple)\
                                                    for element in some_matrix):
            raise ValueError('Elements of input must be list or tuple.')

        # Checks that the input is a 2X2 matrix
        elif len(some_matrix) != 2:
            raise WrongShapeError('Gate must have shape (2, 2).')

        elif len(some_matrix[0]) != 2:
            raise WrongShapeError('Gate must have shape (2, 2).')

        elif len(some_matrix[0]) != len(some_matrix[-1]):
            raise WrongShapeError('Gate must have shape (2, 2).')

        # Initialize the gate
        self.state = np.array(some_matrix)

        # Checks that the input is unitary
        product_with_conj = np.dot(self.state.conj().T, self.state)
        is_unitary = np.allclose(product_with_conj, np.eye(2))
        if is_unitary == False:
            raise NonUnitaryInputError('Gate must be unitary.')

    def print_state(self):
        print(self.state)



class TensorProduct:
    '''Implements a tensor product as a list of all qubits or all operators. Additionally,
    generates the Kronecker representation of the product.'''

    def __init__(self, *arg):
        self.parts = []

        # Check that arg isn't empty
        if len(arg) == 0:
            raise TypeError('Input must be nonempty.')

        # Check that there are multiple arguments.
        elif len(arg) == 1:
            raise TypeError('Input must contain two or more factors.')

        else:

            # Check that all the args are either Qubits() or Gates().
            if any(not isinstance(argument, type(Qubit())) and not isinstance(argument, type(Gate())) for argument in arg):
                raise TypeError('All arguments must have type Qubit() or Gate().')

            # Check that argument type is homogenous.
            elif not all(isinstance(argument, type(arg[0])) for argument in arg):
                raise InhomogenousInputError('TensorProduct() can only be take on arguments of same type.')

            # Place qubit or gate objects into list.
            self.product_type = type(arg[0])
            for item in arg:
                self.parts.append(item)

        # Assemble the parts into the Kronecker product.
        self.state = np.kron(arg[0].state, arg[1].state)
        for i in range(len(arg) - 2):
            self.state = np.kron(self.state, arg[i + 2].state)

    def print_state(self):
        if self.product_type == type(Gate()):
            print(self.state)

        elif self.product_type == type(Qubit()):
            padding = len(format(len(self.state), 'b')) - 1
            label = format(0, 'b').zfill(padding)
            amplitude = self.state[0]
            print("({0:.2e})|{1}>".format(amplitude, label), end='')

            for i in range(1, len(self.state)):
                label = format(i, 'b').zfill(padding)
                amplitude = self.state[i]
                print(" + ({:.2e})|{}>".format(amplitude, label), end='')

        print('')


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
# q = Qubit((10j + 4, 17j - 1))
# q0.print_state()
# q1.print_state()
# q.print_state()
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
