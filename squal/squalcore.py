import numpy as np
import cmath

# 1: Make a QCSim class which constitutes a simulated Quantum/Classical
#    computer simulation.(DONE) It has a quantum_reg(ister) and a classical_reg(ister), both of
#    arbitrary size.(DONE) The quantum_reg is a list of Qubit objects, and the classical_reg
#    is a list of binary values. Gates are applied with the instr method(DONE).
# 2: Create Measurement method in QCSim class. (DONE)
# 3: Improve structure of module. (DONE)
# 4: Fine tune syntax by writing a front-end API to interact with the simulation.
# 5: Build in standard gates as functions. Write tests for multi-qubit gates.
# 6: Test out 5-Qubit Deutsch-Jozsa Algorithm.
# 7: Generate QASM and Quil descriptions of code.

class Qubit:
    '''Creates a normalized Qubit object out of some_vector.'''

    def __init__(self, init_state = [1, 0]):
        # Checks that input is valid.
        self.validate_state(init_state)

        # Initialize qubit.
        self.state = np.array(init_state)
        self.normalize()
        self.decompose_into_comp_basis()

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
            raise NullVectorError('State cannot be the null vector.')

        # Checks that some_vector has length 2n with integer n.
        elif len(some_vector) % 2 != 0:
            raise WrongShapeError('Input state must have even length.')

    def change_state(self, new_state):
        # Checks that input is valid.
        self.validate_state(new_state)

        # Changes the state.
        self.state = np.array(new_state)
        self.normalize()
        self.decompose_into_comp_basis()

    def normalize(self):
        dual_state = np.conjugate(self.state)
        norm = np.sqrt(np.dot(self.state, dual_state))
        self.state = np.multiply(1/norm, self.state)
        self.decompose_into_comp_basis()

    def decompose_into_comp_basis(self):
        # Generates a dict with basis state labels as keys and amplitudes as values
        self.computational_decomp = {}
        padding = len(format(len(self.state), 'b')) - 1
        label = format(0, 'b').zfill(padding)
        amplitude = self.state[0]
        self.computational_decomp[label] = amplitude

        for i in range(1, len(self.state)):
            label = format(i, 'b').zfill(padding)
            amplitude = self.state[i]
            self.computational_decomp[label] = amplitude

    def __len__(self):
        # Note that this returns the number of qubits that the given Qubit object
        # corresponds to, not the number of components its vector representation has
        return int(np.log2(len(self.state)))

    def __repr__(self):
        return str(self.state)

    def __str__(self):
        # Generates a string representation of the state in the computational basis
        first_term_flag = 0
        state_rep = ""
        for state_label in self.computational_decomp:
            if first_term_flag == 0:
                state_rep += "({0:.2e})|{1}>".format(self.computational_decomp[state_label], state_label)
                first_term_flag = 1

            elif first_term_flag == 1:
                state_rep += " + ({:.2e})|{}>".format(self.computational_decomp[state_label], state_label)

        return state_rep

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

    def __len__(self):
        # Note that this returns the number of qubits the gate acts on, NOT the
        # size of matrix representation
        return int(np.log2(self.shape[0]))

    def __repr__(self):
        return str(self.state)

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

        # Check that there are no duplicate register locations for the instruction
        if len(q_reg) != len(set(q_reg)):
            raise ValueError('Specified quantum register locations must be unique.')

        # Check that the gate size matches the number of quantum_reg locations
        if len(gate) != len(q_reg):
            raise WrongShapeError('Number of registers must match number of qubits gate operates on.')

        # Check that all the register locations are nonnegative integers
        for location in q_reg:
            if not isinstance(location, int):
                raise TypeError('Quantum register locations must be integer.')

            if location < 0:
                raise ValueError('Quantum register locations must be nonnegative.')

        # If any of the specified quantum_reg locations have not yet been initialized,
        # initialize them (as well as intermediate reg locs) in the |0> state
        if any(q_reg) > self.q_size - 1:
            n_new_registers = max(q_reg) - (self.q_size - 1)
            new_register = Qubit()
            for i in range(n_new_registers - 1):
                new_register = new_register.qubit_product(Qubit())

            self.quantum_reg = new_register.qubit_product(self.quantum_reg)
            self.q_size = len(self.quantum_reg)

        # Swaps operational qubits into order specified by q_reg args by first
        # generating pairs of swaps to make (using a set to avoid duplicates)
        q_flag = 0
        swap_pairs = set()
        for reg_loc in q_reg:
            swap_pairs.add((q_flag, reg_loc))
            q_flag += 1

        # Remove duplicates
        swap_pairs = set((a,b) if a<=b else (b,a) for a,b in swap_pairs)

        # Perform swaps of qubits into the operational order (first specified qubit
        # in q_reg_loc 0, second in q_reg_loc 1, etc.)
        for swap_pair in swap_pairs:
            self.swap(swap_pair[0], swap_pair[1])

        # If the gate and size of the quantum register match, just operate with the gate
        if self.q_size == len(gate):
            operator = gate

        # If the register size is larger, we need to raise the gate (I^n tensored with gate,
        # since operational order means the operational qubits are ordered into the lowest
        # q_reg_locs)
        elif self.q_size > len(gate):
            left_eye = I
            for i in range(self.q_size - len(gate) - 1):
                left_eye = left_eye.gate_product(I)

            operator = left_eye.gate_product(gate)

        self.quantum_reg.state = np.dot(operator.state, self.quantum_reg.state)
        self.quantum_reg.normalize()
        self.quantum_reg.decompose_into_comp_basis()

        # Swap qubits back into original order
        for swap_pair in swap_pairs:
            self.swap(swap_pair[0], swap_pair[1])

    def swap(self, i, j):
        # Swaps higher of i and j by taking the upper index down abs(i - j) times,
        # and then taking the lower index up abs(i - j) - 1 times
        if not isinstance(i, int) or not isinstance(j, int):
            raise TypeError('Input must be integer-valued.')

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
        # Raise ValueError if swap indicies reference location in the quantum_reg
        # that doesn't exist
        if simple_lower < 0 or simple_lower > (self.q_size - 1):
            raise ValueError("One or more register locations specified in swap doesn't exist.")

        if simple_upper < 0 or simple_upper > (self.q_size - 1):
            raise ValueError("One or more register locations specified in swap doesn't exist.")

        # Note that lower index corresponds to right-hand factors, upper index to left-hand
        number_right_eye = int(simple_lower)
        number_left_eye = int(self.q_size - simple_upper - 1)

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

    def measure(self, q_reg_loc, c_reg_loc=''):
        # Performs a measurement on the qubit at q_reg_loc in the quantum_reg,
        # and optionally stores the result at a specified location in the
        # classical_reg

        # First some sanity checks on the input
        if not isinstance(q_reg_loc, int):
            raise TypeError('Quantum register location must be integer-valued.')

        if not isinstance(c_reg_loc, int) and c_reg_loc != '':
            raise TypeError('Classical register location must be integer-valued.')

        if q_reg_loc < 0:
            raise ValueError('Quantum register location must be nonnegative.')

        if c_reg_loc != '':
            if c_reg_loc < 0:
                raise ValueError('Classical register location must be nonnegative.')

        # Now we determine the result of the measurement from amplitudes in
        # the computation decomposition that correspond to the qubit at q_reg_loc
        # being either zero or one
        amplitudes_for_zero = []
        amplitudes_for_one = []

        for state_label in self.quantum_reg.computational_decomp:
            if int(state_label[-1 - q_reg_loc]) == 0:
                amplitudes_for_zero.append(self.quantum_reg.computational_decomp[state_label])

            if int(state_label[-1 - q_reg_loc]) == 1:
                amplitudes_for_one.append(self.quantum_reg.computational_decomp[state_label])

        # We then use the sorted amplitudes to generate the probabilities of either
        # measurement outcome
        prob_for_zero = 0
        prob_for_one = 0

        for amplitude in amplitudes_for_zero:
            prob_for_zero += amplitude * amplitude.conjugate()

        for amplitude in amplitudes_for_one:
            prob_for_one += amplitude * amplitude.conjugate()

        # Check that total probability remains unity
        prob_total = prob_for_zero + prob_for_one
        mach_eps = np.finfo(type(prob_total)).eps
        if not cmath.isclose(prob_total, 1, rel_tol=10*mach_eps):
            raise NormalizationError('Sum over outcome probabilities = {}.'.format(prob_total))

        measurement = np.random.choice(2, p=[prob_for_zero, prob_for_one])

        # Optionally store this measurement result in the classical_reg
        if c_reg_loc != '':
            # Fills in classical_reg with intermediary zeroes if the c_reg_loc
            # hasn't yet been initialized
            if c_reg_loc > (len(self.classical_reg) - 1):
                difference = c_reg_loc - (len(self.classical_reg) - 1)
                for i in range(difference):
                    self.classical_reg.append(0)

            self.classical_reg[c_reg_loc] = measurement

        # Next we project the state of quantum_reg onto the eigenbasis corresponding
        # to the measurement result with the corresponding projection operator,
        # which only has nonzero components on the diagonal when represented in
        # the computational basis
        projector_diag = []
        for state_label in self.quantum_reg.computational_decomp:
            if int(state_label[-1 - q_reg_loc]) == measurement:
                projector_diag.append(1)

            else:
                projector_diag.append(0)

        projector_operator = np.diag(projector_diag)
        new_state = np.dot(projector_operator, self.quantum_reg.state).tolist()
        # The change_state() method automatically normalizes new_state
        self.quantum_reg.change_state(new_state)

# Custom errors
class WrongShapeError(ValueError):
    pass

class NullVectorError(ValueError):
    pass

class NormalizationError(ValueError):
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
