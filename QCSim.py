import numpy as np
# 0: Clean up and improve unit tests.
# 1: Make a QCSim class which constitutes a simulated Quantum/Classical
#    computer simulation.(DONE) It has a quantum_reg(ister) and a classical_reg(ister), both of
#    arbitrary size.(DONE) The quantum_reg is a list of Qubit objects, and the classical_reg
#    is a list of binary values. Gates are applied with the instr method(DONE).
# 3: Create Measurement method in QCSim class.
# 4: Fine tune syntax.
# 5: Build in standard gates and states. Write tests for multi-qubit gates.
# 6: Test out 5-Qubit Deutsch-Jozsa Algorithm.
# 7: Improve structure of module.

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

class Gate:
    '''Creates a unitary gate out of some_matrix.'''

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
            return Gate(new_gate.tolist())

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

# class QuantumRegister(Qubit):
#     '''Data structure for organizing Qubit objects into a working memory.
#      inherits from Qubit class.'''
#
#     def __init__(self, *qubits):
#         self.memory = []
#         self.state = None
#         for qubit in qubits:
#             self.memory.append(qubit)
#
#         self.make_state()
#
#     def add_qubit(self, qubit):
#         self.memory.append(qubit)
#
#     def swap(self, i, j):
#         if i == j:
#             return None
#
#         if i < 0 or j < 0:
#             raise ValueError('Register locations must be positive.')
#
#         if (not isinstance(i, type(int))) or (not isinstance(j, type(int))):
#             raise ValueError('Register locations must be integer.')
#
#         temp_qubit_1 = self.memory[i]
#         temp_qubit_2 = self.memory[j]
#
#         self.memory[i] = temp_qubit_2
#         self.memory[j] = temp_qubit_1
#
#     def update_qubits(self):
#         pass
#
#     def make_state(self):
#         new_state = self.memory[0]
#         for i in range(1, len(self.memory)):
#             new_state = new_state.qubit_product(self.memory[i])
#
#         self.state = Qubit(new_state)
#         self.update_qubits()


class QCSim:
    '''Simulation of the action of a quantum/classical computer with arbitrary memory.'''

    def __init__(self):
        # Initalizes with one Qubit in the |0> state in the zeroeth
        # quantum_reg location, and a 0 in the zeroeth classical_reg location
        self.quantum_reg = Qubit()
        self.classical_reg = [0]
        self.q_size = 1
        self.c_size = 1

    def instr(self, gate, *q_reg):
        # Check that at least one quantum_reg location is specified for the instruction
        if len(q_reg) == 0:
            raise TypeError('One or more quantum register locations must be specified.')
        # Check that the gate size matches the number of quantum_reg locations
        if gate.shape[0] != 2 * len(q_reg):
            raise WrongShapeError('Number of registers must match size of gate.')
        # If any of the specified quantum_reg locations have not yet been initialized,
        # initialize them (as well as intermediate reg locs) in the |0> state
        if any(q_reg) > self.q_size - 1:
            n_new_registers = max(q_reg) - (self.q_size - 1)
            new_register = Qubit()
            for i in range(n_new_registers - 1):
                # self.quantum_reg = self.quantum_reg.qubit_product(Qubit())
                new_register = new_register.qubit_product(Qubit())
            self.quantum_reg = new_register.qubit_product(self.quantum_reg)

            self.q_size = int(np.log2(len(self.quantum_reg.state)))

        # Swaps operational qubits into order specified by q_reg args
        q_flag = 0
        for reg_loc in q_reg:
            self.swap(q_flag, reg_loc)
            q_flag += 1

        # If the gate and size of the quantum register match, just operate with the gate
        if self.q_size == np.log2(gate.shape[0]):
            operator = gate

        # If the register size is larger, we need to raise the gate (I^n kron gate)
        elif self.q_size > np.log2(gate.shape[0]):
            left_eye = I
            for i in range(self.q_size - int(np.log2(gate.shape[0])) - 1):
                left_eye = left_eye.gate_product(I)

            operator = left_eye.gate_product(gate)

        self.quantum_reg.state = np.dot(operator.state, self.quantum_reg.state)

        q_flag = 0
        for reg_loc in q_reg:
            self.swap(q_flag, reg_loc)

    def swap(self, i, j):
        # Swaps higher of i and j by taking the upper index down abs(i - j) times,
        # and then taking the lower index up abs(i - j) - 1 times
        if i == j:
            return None

        upper_index = max(i, j)
        lower_index = min(i, j)
        difference = abs(i - j)

        # Bring upper down with difference elementary swaps (left prepends SWAP, right post)
        for k in range(difference):
            # Indicies for elementary swap on kth loop
            simple_lower = upper_index - (1 + k)
            simple_upper = upper_index - k

            self.elementary_swap(simple_lower, simple_upper)

        # Bring lower up with (difference - 1) elementary swaps (left prepends SWAP, right post)
        for k in range(difference - 1):
            # Indicies for elementary swap on kth loop
            simple_lower = lower_index + (1 + k)
            simple_upper = lower_index + (2 + k)

            self.elementary_swap(simple_lower, simple_upper)

    def elementary_swap(self, simple_lower, simple_upper):
        number_left_eye = int(simple_lower)
        number_right_eye = int(self.q_size - simple_upper - 1)

        if number_left_eye > 0 and number_right_eye > 0:
            # Prep identity factors
            left_eye = I.gate_product(*[I for l in range(number_left_eye - 1)])
            right_eye = I.gate_product(*[I for l in range(number_right_eye - 1)])

            raised_SWAP = left_eye.gate_product(SWAP, right_eye)
            self.quantum_reg.state = np.dot(raised_SWAP.state, self.quantum_reg.state)
            self.quantum_reg = Qubit(self.quantum_reg.state.tolist())

        elif number_left_eye > 0 and number_right_eye == 0:
            # Prep identity factors
            left_eye = I.gate_product(*[I for l in range(number_left_eye - 1)])

            raised_SWAP = left_eye.gate_product(SWAP)
            self.quantum_reg.state = np.dot(raised_SWAP.state, self.quantum_reg.state)
            self.quantum_reg = Qubit(self.quantum_reg.state.tolist())

        elif number_left_eye == 0 and number_right_eye > 0:
            # Prep identity factors
            right_eye = I.gate_product(*[I for l in range(number_right_eye - 1)])

            raised_SWAP = SWAP.gate_product(right_eye)
            self.quantum_reg.state = np.dot(raised_SWAP.state, self.quantum_reg.state)
            self.quantum_reg = Qubit(self.quantum_reg.state.tolist())

        elif number_left_eye == 0 and number_right_eye == 0:
            raised_SWAP = SWAP
            self.quantum_reg.state = np.dot(raised_SWAP.state, self.quantum_reg.state)
            self.quantum_reg = Qubit(self.quantum_reg.state.tolist())

    def measure(self):
        pass

    def print_state(self):
        pass


# Custom errors.
class WrongShapeError(ValueError):
    pass

class NullVectorError(ValueError):
    pass

class InhomogenousInputError(TypeError):
    pass

class NonUnitaryInputError(ValueError):
    pass

# Default Gates
I = Gate()
X = Gate([[0, 1],
          [1, 0]])
Y = Gate([[0, -1j],
           [1j, 0]])
Z = Gate([[1, 0],
          [0, -1]])
H = Gate([[1/np.sqrt(2), 1/np.sqrt(2)],
          [1/np.sqrt(2), -1/np.sqrt(2)]])

SWAP = Gate([[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]])

# Tests

qc = QCSim()

qc.instr(I, 2)
qc.quantum_reg.print_state()
qc.instr(H, 0)
qc.instr(Z, 0)
qc.quantum_reg.print_state()



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
