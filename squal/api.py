import numpy as np
import copy
import sys
import traceback
import cmath

from squal.squalcore import Qubit, Gate
import squal.gates as gt
import squal.errors as sqerr

'''
An integrated platform for simulating quantum operations is provided with the
QCSim class, and a data structure for writing programs which can be executed by it is
provided with the Program class.
'''

class QCSim:
    '''
    Simulates of the action of a quantum/classical computer with arbitrary memory.
    '''

    def __init__(self):
        # Initalizes with one Qubit in the |0> state in the zeroeth
        # quantum_reg location, and a 0 in the zeroeth classical_reg location
        self.__quantum_reg = Qubit()
        self.__classical_reg = [0]
        self.__q_size = 1
        self.__c_size = 1
        self.__declared_gates = []

    def __instr(self, gate, *q_reg, kraus_ops=None):
        # If the list kraus_ops is provided, then the noisy quantum operation that
        # those Kraus operators defines are applied. Otherwise, the instruction isn't
        # noisy.

        # Check that at least one quantum_reg location is specified for the instruction
        if len(q_reg) == 0:
            raise TypeError('One or more quantum register locations must be specified.')

        # Check that there are no duplicate register locations for the instruction
        if len(q_reg) != len(set(q_reg)):
            raise ValueError('Specified quantum register locations must be unique.')

        # Check that the gate size matches the number of quantum_reg locations
        if len(gate) != len(q_reg):
            raise sqerr.WrongShapeError('Number of registers must match number of qubits gate operates on.')

        # Check that all the register locations are nonnegative integers
        for location in q_reg:
            if not isinstance(location, int):
                raise TypeError('Quantum register locations must be integer.')

            if location < 0:
                raise ValueError('Quantum register locations must be nonnegative.')

        # Check that the gate is either a standard gate or has been previously declared.
        # Nested loop checks for rot and phase gates.
        if (not any(gate.name() == std_gate for std_gate in gt.STD_GATES)) and\
            (not any(gate.name() == usr_gate for usr_gate in self.__declared_gates)):
                if (not any(gate.name()[:2] == std_gate for std_gate in gt.STD_GATES)) and\
                    (not any(gate.name()[:4] == std_gate for std_gate in gt.STD_GATES)):
                        raise sqerr.UndeclaredGateError("Unknown Gate.")

        # If any of the specified quantum_reg locations have not yet been initialized,
        # initialize them (as well as intermediate reg locs) in the |0> state
        if max(q_reg) > self.__q_size - 1:
            n_new_registers = max(q_reg) - (self.__q_size - 1)
            new_register = Qubit()
            for i in range(n_new_registers - 1):
                new_register = new_register.qubit_product(Qubit())

            self.__quantum_reg = new_register.qubit_product(self.__quantum_reg)
            self.__q_size = len(self.__quantum_reg)

        # Initialize an identity gate for later use (and throw away target qubit)
        I = gt.I(0)[0]

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
            self.__swap(swap_pair[0], swap_pair[1])

        # If the gate and size of the quantum register match, just operate with the gate
        if self.__q_size == len(gate):
            operator = gate

        # If the register size is larger, we need to raise the gate (I^n tensored with gate,
        # since operational order means the operational qubits are ordered into the lowest
        # q_reg_locs)
        elif self.__q_size > len(gate):
            left_eye = I
            for i in range(self.__q_size - len(gate) - 1):
                left_eye = left_eye.gate_product(I)

            operator = left_eye.gate_product(gate)

        # If no Kraus operators are specified, evaluation of new register state is simple.
        if kraus_ops == None:
            new_reg_state = np.dot(operator.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_reg_state.tolist())

        else:
            # First we randomly choose one of the Kraus operators to apply, then we
            # generate a corresponding gate to hand over to the __instr() method.
            current_state = np.dot(operator.state(), self.__quantum_reg.state())
            probs = []
            new_state_ensemble = []

            # Generate an ensemble of states transformed according to Kraus ops in the
            # form of a list of transformed state vector, and a corresponding
            # list of weights for each tranformation
            for op in kraus_ops:
                # Raise each operator if necessary
                if self.__q_size > np.log2(op.shape[0]):
                    k = np.kron(left_eye.state(), op)
                else:
                    k = op
                new_state = k.dot(current_state)
                new_state_ensemble.append(new_state)
                new_dual = np.conjugate(new_state)
                probability = np.dot(new_state, new_dual)
                probs.append(probability)

            # Pick one of the transformed states according to probs
            new_state_index = np.random.choice([i for i in range(len(new_state_ensemble))], p=probs)
            new_reg_state = new_state_ensemble[new_state_index]
            self.__quantum_reg.change_state(new_reg_state.tolist())

        # Swap qubits back into original order
        for swap_pair in swap_pairs:
            self.__swap(swap_pair[0], swap_pair[1])

    def __cinstr(self, name, *target_bits):
        # Checks that name is a string and that a valid target is specified
        if len(target_bits) == 0:
            raise ValueError('No target specified for classical instruction.')

        if not isinstance(name, str):
            raise ValueError('The name of the classical instruction must be a string.')

        if name not in gt.CLASSICAL_OPS:
            raise ValueError('Unknown operation.')

        # If any of the target_bits are outside the range of the classical register,
        # initialize intermediary bits with the value zero
        if max(target_bits) > len(self.__classical_reg) - 1:
            diff = max(target_bits) - (len(self.__classical_reg) - 1)
            for i in range(diff):
                self.__classical_reg.append(0)

            self.__c_size = len(self.__classical_reg)

        # We handle COPY and EXCHANGE operations separately
        if name != 'COPY' and name != 'EXCHANGE':
            # The last argument in target_bits is the c_reg_index where the result of
            # calling either of a unary or binary operation on input_values gets stored
            input_values = [self.__classical_reg[target] for target in target_bits]
            if len(input_values) > 1:
                input_values.pop()

            c_reg_index = target_bits[-1]
            new_bit_value = gt.CLASSICAL_OPS[name](*input_values)
            self.__classical_reg[c_reg_index] = new_bit_value

        if name == 'COPY' or name == 'EXCHANGE':
            if len(target_bits) != 2:
                raise ValueError('COPY and EXCHANGE operations require 2 targets')

            else:
                c_reg_index_1 = target_bits[0]
                c_reg_index_2 = target_bits[1]
                input_values = [self.__classical_reg[target] for target in target_bits]

                new_bit_values = gt.CLASSICAL_OPS[name](*input_values)

                self.__classical_reg[c_reg_index_1] = new_bit_values[0]
                self.__classical_reg[c_reg_index_2] = new_bit_values[1]

    def __swap(self, i, j):
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

            self.__elementary_swap(simple_lower, simple_upper)

        # Bring lower up with (difference - 1) elementary swaps (left prepends SWAP, right post)
        for k in range(difference - 1):
            # Indicies for elementary swap on kth loop
            simple_lower = lower_index + (1 + k)
            simple_upper = lower_index + (2 + k)

            self.__elementary_swap(simple_lower, simple_upper)

    def __elementary_swap(self, simple_lower, simple_upper):
        # Raise ValueError if swap indicies reference location in the quantum_reg
        # that doesn't exist
        if simple_lower < 0 or simple_lower > (self.__q_size - 1):
            raise ValueError("One or more register locations specified in swap doesn't exist.")

        if simple_upper < 0 or simple_upper > (self.__q_size - 1):
            raise ValueError("One or more register locations specified in swap doesn't exist.")

        # Initialize identity and swap gates for later use (and throw away target qubit)
        I = gt.I(0)[0]
        SWAP = gt.SWAP(0, 1)[0]

        # Note that lower index corresponds to right-hand factors, upper index to left-hand
        number_right_eye = int(simple_lower)
        number_left_eye = int(self.__q_size - simple_upper - 1)

        if number_left_eye > 0 and number_right_eye > 0:
            # Prep identity factors
            left_eye = I.gate_product(*[I for l in range(number_left_eye - 1)])
            right_eye = I.gate_product(*[I for l in range(number_right_eye - 1)])

            raised_SWAP = left_eye.gate_product(SWAP, right_eye)
            new_swap_state = np.dot(raised_SWAP.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_swap_state.tolist())
            # self.quantum_reg.state = np.dot(raised_SWAP.state, self.quantum_reg.state)
            # self.quantum_reg = Qubit(self.quantum_reg.state.tolist())

        elif number_left_eye > 0 and number_right_eye == 0:
            # Prep identity factors
            left_eye = I.gate_product(*[I for l in range(number_left_eye - 1)])

            raised_SWAP = left_eye.gate_product(SWAP)
            new_swap_state = np.dot(raised_SWAP.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_swap_state.tolist())
            # self.quantum_reg.state = np.dot(raised_SWAP.state, self.quantum_reg.state)
            # self.quantum_reg = Qubit(self.quantum_reg.state.tolist())

        elif number_left_eye == 0 and number_right_eye > 0:
            # Prep identity factors
            right_eye = I.gate_product(*[I for l in range(number_right_eye - 1)])

            raised_SWAP = SWAP.gate_product(right_eye)
            new_swap_state = np.dot(raised_SWAP.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_swap_state.tolist())
            # self.quantum_reg.state = np.dot(raised_SWAP.state, self.quantum_reg.state)
            # self.quantum_reg = Qubit(self.quantum_reg.state.tolist())

        elif number_left_eye == 0 and number_right_eye == 0:
            raised_SWAP = SWAP
            new_swap_state = np.dot(raised_SWAP.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_swap_state.tolist())
            # self.quantum_reg.state = np.dot(raised_SWAP.state, self.quantum_reg.state)
            # self.quantum_reg = Qubit(self.quantum_reg.state.tolist())

    def __measure(self, q_reg_loc, c_reg_loc=''):
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

        if q_reg_loc > self.__q_size - 1:
            raise ValueError('Specified quantum register location does not exist.')

        if c_reg_loc != '':
            if c_reg_loc < 0:
                raise ValueError('Classical register location must be nonnegative.')

        # Now we determine the result of the measurement from amplitudes in
        # the computation decomposition that correspond to the qubit at q_reg_loc
        # being either zero or one
        amplitudes_for_zero = []
        amplitudes_for_one = []

        basis_states = self.__quantum_reg.computational_decomp()
        for state_label in basis_states:
            if int(state_label[-1 - q_reg_loc]) == 0:
                amplitudes_for_zero.append(basis_states[state_label])

            if int(state_label[-1 - q_reg_loc]) == 1:
                amplitudes_for_one.append(basis_states[state_label])

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
            raise sqerr.NormalizationError('Sum over outcome probabilities = {}.'.format(prob_total))

        measurement = np.random.choice(2, p=[prob_for_zero, prob_for_one])

        # Optionally store this measurement result in the classical_reg
        if c_reg_loc != '':
            # Fills in classical_reg with intermediary zeroes if the c_reg_loc
            # hasn't yet been initialized
            if c_reg_loc > (len(self.__classical_reg) - 1):
                difference = c_reg_loc - (len(self.__classical_reg) - 1)
                for i in range(difference):
                    self.__classical_reg.append(0)

            self.__classical_reg[c_reg_loc] = measurement

        # Next we project the state of quantum_reg onto the eigenbasis corresponding
        # to the measurement result with the corresponding projection operator,
        # which only has nonzero components on the diagonal when represented in
        # the computational basis
        projector_diag = []
        for state_label in basis_states:
            if int(state_label[-1 - q_reg_loc]) == measurement:
                projector_diag.append(1)

            else:
                projector_diag.append(0)

        projector_operator = np.diag(projector_diag)
        new_state = np.dot(projector_operator, self.__quantum_reg.state())
        # Note that the change_state() method automatically normalizes new_state
        self.__quantum_reg.change_state(new_state.tolist())

    def __if(self, test_loc, then_branch, else_branch = None):
        '''
        Uses the contents of the classical register at the test location to
        evaluate either then_branch, or else_branch (if specified).
        '''

        # First we do sanity checks on the specified test index
        if not isinstance(test_loc, int):
            raise TypeError("Classical register test index must be integer.")

        if test_loc > len(self.__classical_reg) - 1 or test_loc < 0:
            raise ValueError("Classical register test location out of range.")

        # Now the logic of the if/else instructions
        if self.__classical_reg[test_loc] == 1:
            for line in then_branch:
                self.__elementary_eval(line)

        elif else_branch != None:
            for line in else_branch:
                self.__elementary_eval(line)

        else:
            pass


    def __while(self, test_loc, body):
        '''
        Uses the contents of the classical register at the test location to
        control loop.
        '''

        # First we do sanity checks on the specified test index
        if not isinstance(test_loc, int):
            raise TypeError("Classical register test index must be integer.")

        if test_loc > len(self.__classical_reg) - 1 or test_loc < 0:
            raise ValueError("Classical register test location out of range.")

        while self.__classical_reg[test_loc] == 1:
            for line in body:
                self.__elementary_eval(line)

    def __reset(self):
        '''
        Resets the quantum and classical registers.
        '''
        self.__quantum_reg = Qubit()
        self.__classical_reg = [0]
        self.__q_size = 1
        self.__c_size = 1
        self.__declared_gates = []

    def __elementary_eval(self, line):
        '''
        Primitive method for handing over a single instruction for execution.
        '''

        # If the line is a quantum instruction (Gate, *targets), hand the
        # gate and targets over to self.__instr()
        if isinstance(line[0], type(Gate())):
            self.__instr(line[0], *line[1:])

        # For a NEWGATE instruction ('NEWGATE', gate_name, gates), we add the gate
        # to the list of user declared gates.
        elif line[0] == 'NEWGATE':
            self.__declared_gates.append(line[1])

        # If the line is a measurement instruction
        # ('MEASURE', q_reg_loc, optional c_reg_loc), hand the
        # contents over to self.__measure()
        elif line[0] == 'MEASURE':
            q_loc = line[1]
            c_loc = ''
            if len(line) == 3:
                c_loc = line[2]

            self.__measure(q_loc, c_loc)

        # Noise instructions also hand a list of Kraus operators to self.__instr()
        elif line[0] == 'NOISY':
            # line[1] is a list of Kraus operators as np arrays,
            # line[2] is a gate_target_tuple
            self.__instr(line[2][0], *line[2][1:], kraus_ops=line[1])

        elif line[0] == 'WHILE':
            test_loc = line[1]
            body = line[2]
            self.__while(test_loc, body)

        elif line[0] == 'IF':
            # Gets the classical register test location
            test_loc = line[1]
            # This branch happens when no 'else' is specified
            if len(line) == 3:
                then_branch = line[2]
                self.__if(test_loc, then_branch)

            # This branch only happens when an 'else' is specified
            if len(line) == 4:
                then_branch = line[2]
                else_branch = line[3]
                self.__if(test_loc, then_branch, else_branch = else_branch)

            # Catch-all for malform 'IF' instructions since it should have
            # length of either 3 or 4
            elif len(line) != 3 and len(line) != 4:
                raise TypeError("Improper syntax in 'IF' statement.")

        elif line[0] in gt.CLASSICAL_OPS:
            self.__cinstr(line[0], *line[1:])

        else:
            raise sqerr.UndeclaredGateError("Unknown Gate.")

    def execute(self, program):
        '''
        Returns the contents of the classical register after executing a program.
        '''
        if not isinstance(program, type(Program())):
            raise TypeError('Can only execute Program objects.')

        line_number = 0
        # Run through each line of the program
        for program_line in program:
            line_number += 1
            try:
                self.__elementary_eval(program_line)
            except Exception as ex:
                # Generate error message
                message = ""
                message += type(ex).__name__
                message += " occcured on program line {}: ".format(line_number)
                message += "{}".format(program_line)
                print(message)
                print(traceback.format_exc())
                sys.exit(1)

        output_c_reg = copy.deepcopy(self.__classical_reg)

        self.__reset()

        return output_c_reg

    def quantum_state(self, program):
        '''
        Returns the state of the quantum register after executing a program.
        From a physical perspective, this is cheating!
        '''
        if not isinstance(program, type(Program())):
            raise TypeError('Can only execute Program objects.')

        line_number = 0
        # Run through each line of the program
        for program_line in program:
            line_number += 1
            try:
                self.__elementary_eval(program_line)
            except Exception as ex:
                # Generate error message
                message = ""
                message += type(ex).__name__
                message += " occcured on program line {}: ".format(line_number)
                message += "{}".format(program_line)
                print(message)
                print(traceback.format_exc())
                sys.exit(1)

        output_q_reg = Qubit(self.__quantum_reg.state().tolist())

        self.__reset()

        return output_q_reg

class Program():
    '''
    Program class provides a data structure for composing and organizing programs
    to run on QCSim.
    '''

    def __init__(self):
        self.__instructions = []

    def __iter__(self):
        self.__line = -1
        return self

    def __next__(self):
        self.__line += 1

        if self.__line > len(self) - 1:
            raise StopIteration
        return self.__instructions[self.__line]

    def add_instr(self, gate_target_tuple, position=None):
        '''
        Adds a quantum instruction to self.__instructions, with the default behavior
        being to append the instruction.
        '''

        # Sets default appending behavior
        if position == None:
            position = len(self)

        if position > len(self.__instructions) or position < 0 or not isinstance(position, int):
            raise ValueError('Invalid program position number. Out of range.')

        if not isinstance(gate_target_tuple, tuple):
            raise TypeError('Argument must be a tuple of Gate object followed by target qubits.')

        elif not isinstance(gate_target_tuple[0], type(Gate())):
            raise TypeError('First element of argument must be a Gate object.')

        for i in range(1, len(gate_target_tuple)):
            if not isinstance(gate_target_tuple[i], int) or gate_target_tuple[i] < 0:
                raise ValueError('Target qubits must be nonnegative integers.')

        self.__instructions.insert(position, gate_target_tuple)

    def add_ninstr(self, gate_target_tuple, kraus_ops, position=None):
        # Generates an instruction ('NOISY', kraus_ops, gate_target_tuple).
        # The compiler handles execution of the 'NOISY' instruction by computing
        # the probabilities corresponding to each Kraus operator matrix in the
        # kraus_ops list. Unit test that sum E^{dagger}E <= 1.

        # Sets default appending behavior
        if position == None:
            position = len(self)

        # Checks that the position is valid
        if position > len(self.__instructions) or position < 0 or not isinstance(position, int):
            raise ValueError('Invalid program position number. Out of range.')

        # Check that kraus_ops is a list
        if not isinstance(kraus_ops, list):
            raise TypeError("The argument kraus_ops must be a nonempty list.")

        # Check that kraus_ops has at least two elements
        if len(kraus_ops) < 2:
            raise ValueError("The list kraus_ops must have at least two elements.")

        # Check that each element of kraus_ops is a ndarray
        if not all(isinstance(op, type(np.array([]))) for op in kraus_ops):
            raise TypeError("Each operator in kraus_ops must be a numpy array.")

        # Check that each operator in kraus_ops has the correct shape.
        kraus_shape = kraus_ops[0].shape
        if kraus_shape[0] != kraus_shape[1]:
            raise sqerr.WrongShapeError("Kraus operators must be square matricies.")

        for op in kraus_ops:
            if op.shape != kraus_shape:
                raise sqerr.WrongShapeError("Kraus operator shapes must be homogeneous.")

        # Check that the size of the Kraus operators agrees with the gate size
        gate_shape = gate_target_tuple[0].state().shape
        if gate_shape != kraus_shape:
            raise sqerr.WrongShapeError("Size mismatch between Kraus operators and gate.")


        # Check that kraus_ops satisfy completeness
        identity = np.identity(kraus_shape[0])
        sum = np.matmul(np.conjugate(kraus_ops[0].T), kraus_ops[0])
        for i in range(1, len(kraus_ops)):
            sum += np.matmul(np.conjugate(kraus_ops[i].T), kraus_ops[i])

        if not np.allclose(sum, identity):
            raise sqerr.NormalizationError("Kraus operators must satisfy completeness relation.")

        if not isinstance(gate_target_tuple, tuple):
            raise TypeError('gate_target_tuple must be a tuple of Gate object followed by target qubits.')

        elif not isinstance(gate_target_tuple[0], type(Gate())):
            raise TypeError('First element of gate_target_tuple must be a Gate object.')

        for i in range(1, len(gate_target_tuple)):
            if not isinstance(gate_target_tuple[i], int) or gate_target_tuple[i] < 0:
                raise ValueError('Target qubits must be nonnegative integers.')

        self.__instructions.insert(position, ('NOISY', kraus_ops, gate_target_tuple))

    def add_cinstr(self, classical_gate, position=None):
        '''
        Adds a classical unary or binary instruction to self.__instructions, with
        the default behavior being to append the instruction.
        '''

        # Sets default appending behavior
        if position == None:
            position = len(self)

        if position > len(self.__instructions) or position < 0 or not isinstance(position, int):
            raise ValueError('Invalid program position number. Out of range.')

        if not isinstance(classical_gate, tuple):
            raise TypeError('Argument must be a tuple of gate name string followed by target bits.')

        elif not isinstance(classical_gate[0], str):
            raise TypeError('First element of argument must be a string (name of gate).')

        self.__instructions.insert(position, classical_gate)

    def rm_instr(self, position=None):
        '''
        Removes a generic instruction from self.__instructions by index. The default
        behavior is to remove the last instruction.
        '''

        if position == None:
            position = len(self) - 1

        if position > len(self.__instructions) or position < 0 or not isinstance(position, int):
            raise ValueError('Invalid program position number. Must be a nonnegative integer.')

        del self.__instructions[position]

    def measure(self, qubit_loc, classical_loc=None, position=None):
        '''
        Adds to self.__instructions a special instruction to measure the qubit
        at quantum register location qubit_loc and optionally save it in the
        classical register at the location classical_loc. The default
        behavior is to append the measurement to the end of the program, but
        the instruction can be inserted by setting position to the desired program
        line (zero-indexed).
        '''

        # If the classical_loc isn't valid, throw a ValueError
        if (not isinstance(classical_loc, int) and not isinstance(classical_loc, type(None))):
            raise ValueError('Classical register location must be a nonnegative integer.')

        if not isinstance(classical_loc, type(None)):
            if classical_loc < 0:
                raise ValueError('Classical register location must be a nonnegative integer.')

        # Sets default appending behavior
        if position == None:
            position = len(self)

        # If the program position is out of range, throw a ValueError
        if position > len(self.__instructions) or position < 0 or not isinstance(position, int):
            raise ValueError('Invalid program position number. Out of range.')

        # Branch in instruction depending on if a classical_loc is specified for
        # storing the measurement.
        if classical_loc == None:
            self.__instructions.insert(position, ('MEASURE', qubit_loc))

        if isinstance(classical_loc, int) and classical_loc >= 0:
            self.__instructions.insert(position, ('MEASURE', qubit_loc, classical_loc))

    def if_then_else(self, test_loc, if_branch, else_branch = None):
        '''
        Adds the instruction to execute the subprogram 'if_branch' if the
        bit at classical register index 'test_loc' is 1. Otherwise, if an 'else_branch'
        Program object is specified, that gets executed. If none is specified,
        execution passes on.
        '''

        # First we check that the branches are valid Program objects
        if not isinstance(if_branch, type(Program())):
            raise TypeError("Branches of 'IF' statement must be Program objects.")

        if not isinstance(else_branch, type(Program()))\
            and not isinstance(else_branch, type(None)):
            raise TypeError("Else branch of 'IF' statement, if specified, must be a Program object.")

        # Now we construct the instruction
        if not isinstance(else_branch, type(None)):
            ite_instruction = ('IF', test_loc, if_branch.instructions(),\
                               else_branch.instructions())

        else:
            ite_instruction = ('IF', test_loc, if_branch.instructions())

        self.__instructions.append(ite_instruction)

    def while_loop(self, test_loc, body):
        '''
        Adds the instruction to execute the subprogram 'body' while the
        bit at classical register index 'test_loc' is 1.
        '''

        # First we check that the body is a valid program object
        if not isinstance(body, type(Program())):
            raise TypeError('Body of while statement must be a Program object.')

        loop_instruction = ('WHILE', test_loc, body.instructions())
        self.__instructions.append(loop_instruction)

    def new_gate(self, new_gate_instruction):
        '''
        Defines a new gate within the squal environment and returns a function to
        the user implementing that gate. The instruction takes the form
        ('NEWGATE', gate_name, gates), where each gate in the list gates is assumed
        to act on the same target qubits and are applied in left to right order.
        '''

        # First we check that new_gate_instruction is well-formed.
        if not isinstance(new_gate_instruction, tuple):
            raise TypeError('Argument of new_gate must be a tuple.')

        if not len(new_gate_instruction) == 3:
            raise TypeError('Instruction has the wrong number of terms.')

        if not new_gate_instruction[0] == 'NEWGATE':
            raise ValueError("Instruction must begin with 'NEWGATE' keyword.")

        if len(new_gate_instruction[2]) < 1:
            raise TypeError('NEWGATE must combine at least two gates.')

        if not all(isinstance(gate, type(Gate())) for gate in new_gate_instruction[2]):
            raise TypeError("Third term of 'NEWGATE' instruction must be list of Gate objects.")

        gate_shape = new_gate_instruction[2][0].shape()
        for gate in new_gate_instruction[2]:
            if gate.shape() != gate_shape:
                raise sqerr.WrongShapeError("All gates in 'NEWGATE' must have same shape.")

        # Now we generate the function implementing the new gate.
        matrix_reps = [gate.state() for gate in new_gate_instruction[2]]
        matrix_rep = matrix_reps[0]
        for i in range(1, len(matrix_reps)):
            matrix = matrix_reps[i]
            matrix_rep = np.matmul(matrix, matrix_rep)

        def gate_function(*target_qubits):
            return (Gate(matrix_rep.tolist(), name=new_gate_instruction[1]), *target_qubits)

        # Finally, we append the instruction to the program, and then return
        # gate_function to the user.
        self.__instructions.append(new_gate_instruction)
        return gate_function

    def instructions(self):
        '''
        Helper method for returning a copy of the instructions in list form.
        '''

        return copy.deepcopy(self.__instructions)

    def __len__(self):
        return len(self.__instructions)

    def __repr__(self):
        program_rep = "{\n"
        for instruction in self.__instructions:
            program_rep += str(instruction)
            program_rep += '\n'

        program_rep += '}'
        return program_rep
