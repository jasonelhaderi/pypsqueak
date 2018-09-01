import numpy as np
import copy
import sys
import traceback
import inspect
import cmath

from pypsqueak.squeakcore import Qubit, Gate
import pypsqueak.gates as gt
import pypsqueak.errors as sqerr

'''
API for pypSQUEAK. A hybrid quantum/classical virtual machine is provided by the
qcVirtualMachine class for running circuits composed with the Program class.
The Program class provides a data structure for writing programs to be executed
by the virtual machine.
'''

# Protected keywords and chars.
_keywords = ['MEASURE', 'IF', 'THEN', 'ELSE', 'WHILE', 'DO', 'GATEDEF']
_protected_chars = ['+', '-', '=', '*', '(', ')', ';', ':', '{', '}', '[', ']']

class qcVirtualMachine:
    '''
    Implementation of a hybrid quantum/classical computer. Has both classical and
    and quantum registers of arbitrary size which are implemented as a list
    and a Qubit object, respectively. Additionally, there are four instruction
    registers:
        - self.__declared_gates ~~~~~ User defined quantum gates
        - self.__declared_pgates ~~~~ User defined parametric quantum gates
        - self.__builtin_qgates ~~~~~ Standard set of quantum gates
        - self.__builtin_cgates ~~~~~ Classical logic gates

    In this implementation, matrix representations (in the form of list, tuple,
    or numpy ndarray types) are stored in the registers, with Gate objects only
    being constructed at execution time of a given quantum instruction.
    '''

    def __init__(self):
        '''
        Initalizes the machine with a one qubit quantum register in the |0> state
        and a one bit classical register set to '0'.
        '''

        self.__quantum_reg = Qubit()
        self.__classical_reg = [0]
        self.__q_size = 1
        self.__c_size = 1
        self.__declared_gates = {}
        self.__declared_pgates = {}
        self.__builtin_qgates = {**gt.STD_GATES}
        self.__builtin_cgates = {**gt.CLASSICAL_OPS}

    def __instr(self, gate, *q_reg, kraus_ops=None):
        '''
        Applies a quantum gate to the quantum register. If a list of Kraus operators
        gets handed to the method, then the application is performed as a generalized
        quantum operation (typically, the set of operators would characterize a kind of
        noise). Since the qcVirtualMachine quantum register corresponds to a pure state,
        generalized quantum operations are implemented as follows:
            1) First, the noiseless quantum gate is applied to the quantum register.
            2) Next, each of the Kraus operators gets applied to this new quantum
               register state in order to to generate an ensemble of unnormalized
               outcome states.
            2) The norm of each outcome state is computed and interpreted as a probability.
            3) These probabilities are used as weights to randomly pick one outcome from
               the ensemble of outcomes. This outcome is then normalized and used
               as the ultimate quantum register state.

        Note that only trace-preserving Kraus operations are allowed in this
        implementation.
        '''

        # Check that at least one quantum_reg location is specified for the instruction
        if len(q_reg) == 0:
            raise IndexError('One or more quantum register locations must be specified.')

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

        # Now we run some more checks on the validity of the Kraus operators
        if kraus_ops != None:

            # Check that the size of the Kraus operators agrees with the gate size
            kraus_shape = kraus_ops[0].shape
            gate_shape = gate.state().shape
            if gate_shape != kraus_shape:
                raise sqerr.WrongShapeError("Size mismatch between Kraus operators and gate.")

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
        I = Gate(gt._I)

        # Swap operational qubits into order specified by q_reg args; first we
        # generate pairs of swaps to make (using a set to avoid duplicates)
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

        # If no Kraus operators are specified, evaluation of new register state is trivial
        if kraus_ops == None:
            new_reg_state = np.dot(operator.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_reg_state)

        else:
            # We randomly choose one of the Kraus operators to apply, then we
            # generate a corresponding gate to hand over to the __instr() method
            current_state = np.dot(operator.state(), self.__quantum_reg.state())
            probs = []
            new_state_ensemble = []

            # Generate an ensemble of states transformed according to Kraus ops in the
            # form of a list of transformed state vector, and a corresponding
            # list of probability weights for each tranformation
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
            self.__quantum_reg.change_state(new_reg_state)

        # Swap qubits back into original order
        for swap_pair in swap_pairs:
            self.__swap(swap_pair[0], swap_pair[1])

    def __cinstr(self, name, *target_bits):
        '''
        Applies a classical logic gate to the classical register. If any of the
        targets of the gate are out of the range of the classical register, all
        necessary new bits get initialized to the state '0' before gate application.
        '''

        # Checks that name is a string and that a valid target is specified
        if len(target_bits) == 0:
            raise ValueError('No target specified for classical instruction.')

        if not isinstance(name, str):
            raise ValueError('The name of the classical instruction must be a string.')

        if name not in self.__builtin_cgates:
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
            new_bit_value = self.__builtin_cgates[name](*input_values)
            self.__classical_reg[c_reg_index] = new_bit_value

        if name == 'COPY' or name == 'EXCHANGE':
            if len(target_bits) != 2:
                raise ValueError('COPY and EXCHANGE operations require 2 targets')

            else:
                c_reg_index_1 = target_bits[0]
                c_reg_index_2 = target_bits[1]
                input_values = [self.__classical_reg[target] for target in target_bits]

                new_bit_values = self.__builtin_cgates[name](*input_values)

                self.__classical_reg[c_reg_index_1] = new_bit_values[0]
                self.__classical_reg[c_reg_index_2] = new_bit_values[1]

    def __swap(self, i, j):
        '''
        Method for swapping the ith and jth qubits in the quantum register.
        '''

        # Swaps higher of i and j by taking the upper index down abs(i - j) times,
        # and then taking the lower index up abs(i - j) - 1 times
        if not isinstance(i, int) or not isinstance(j, int):
            raise IndexError('Quantum register indicies must be integer-valued.')

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
        '''
        Helper method to swap adjacent qubits in the quantum register.
        '''

        # Raise IndexError if swap indicies reference location in the quantum_reg
        # that doesn't exist
        if simple_lower < 0 or simple_lower > (self.__q_size - 1):
            raise IndexError("One or more register locations specified in swap doesn't exist.")

        if simple_upper < 0 or simple_upper > (self.__q_size - 1):
            raise IndexError("One or more register locations specified in swap doesn't exist.")

        # Initialize identity and swap gates for later use (and throw away target qubit)
        I = Gate(gt._I)
        SWAP = Gate(gt._SWAP)

        # Note that lower index corresponds to right-hand factors, upper index to left-hand
        number_right_eye = int(simple_lower)
        number_left_eye = int(self.__q_size - simple_upper - 1)

        if number_left_eye > 0 and number_right_eye > 0:
            # Prep identity factors
            left_eye = I.gate_product(*[I for l in range(number_left_eye - 1)])
            right_eye = I.gate_product(*[I for l in range(number_right_eye - 1)])

            raised_SWAP = left_eye.gate_product(SWAP, right_eye)
            new_swap_state = np.dot(raised_SWAP.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_swap_state)

        elif number_left_eye > 0 and number_right_eye == 0:
            # Prep identity factors
            left_eye = I.gate_product(*[I for l in range(number_left_eye - 1)])

            raised_SWAP = left_eye.gate_product(SWAP)
            new_swap_state = np.dot(raised_SWAP.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_swap_state)

        elif number_left_eye == 0 and number_right_eye > 0:
            # Prep identity factors
            right_eye = I.gate_product(*[I for l in range(number_right_eye - 1)])

            raised_SWAP = SWAP.gate_product(right_eye)
            new_swap_state = np.dot(raised_SWAP.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_swap_state)

        elif number_left_eye == 0 and number_right_eye == 0:
            raised_SWAP = SWAP
            new_swap_state = np.dot(raised_SWAP.state(), self.__quantum_reg.state())
            self.__quantum_reg.change_state(new_swap_state)

    def __measure(self, q_reg_loc, c_reg_loc=''):
        '''
        Measures the qubit specified by index q_reg_loc and optionally stores
        the result in the classical register at the location specified by the
        address c_reg_loc.
        '''

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
        self.__quantum_reg.change_state(new_state)

    def __if(self, test_loc, then_branch, else_branch = None):
        '''
        Uses the contents of the classical register at the address specified by
        test_loc to conditionally evaluate either of the Program objects
        then_branch or else_branch (if None, equivalent to an empty statement).
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
        Executes the Program object body until the content of the classical
        register at the address specified by test_loc is '0'.
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
        Resets the quantum, classical, and user-declared instruction registers.
        '''

        self.__quantum_reg = Qubit()
        self.__classical_reg = [0]
        self.__q_size = 1
        self.__c_size = 1
        self.__declared_gates = {}
        self.__declared_pgates = {}

    def __elementary_eval(self, line):
        '''
        Parses a single instruction and calls the appropriate method.
        '''

        # If the line is a quantum/classical gate instruction
        # (gate_target_tuple, params_dict=None, kraus_ops=None),
        # check if the gate name is built in or previously declared, then hand the
        # gate and targets over to self.__instr()
        if isinstance(line[0], tuple):
            gate_name = line[0][0]
            targets = line[0][1:]
            params_dict = line[1]
            k_op_list = line[2]

            # Initialize a Gate object if a quantum gate instruction.
            if gate_name in self.__builtin_qgates:
                if callable(self.__builtin_qgates[gate_name]):
                    gate_obj = Gate(self.__builtin_qgates[gate_name](**params_dict), name = gate_name)
                else:
                    gate_obj = Gate(self.__builtin_qgates[gate_name], name = gate_name)
            elif gate_name in self.__declared_gates:
                gate_obj = Gate(self.__declared_gates[gate_name], name = gate_name)
            elif gate_name in self.__declared_pgates:
                gate_obj = Gate(self.__declared_pgates[gate_name](**params_dict), name = gate_name)
            # If the instruction is a classical gate, simply apply it.
            elif gate_name in self.__builtin_cgates:
                gate_obj = None
            else:
                raise sqerr.UndeclaredGateError("Unknown Gate '{}'.".format(gate_name))

            # Run the instruction.
            if gate_obj == None:
                self.__cinstr(gate_name, *targets)
            else:
                self.__instr(gate_obj, *targets, kraus_ops=k_op_list)

        # For a GATEDEF instruction ('GATEDEF', name, matrix_rep),
        elif line[0] == 'GATEDEF':
            gate_name = line[1]
            matrix_rep = line[2]
            if callable(matrix_rep):
                self.__declared_pgates[gate_name] = matrix_rep
            else:
                self.__declared_gates[gate_name] = matrix_rep

        # If the instruction is a measurement instruction
        # ('MEASURE', q_reg_loc, optional c_reg_loc), hand the
        # contents over to self.__measure()
        elif line[0] == 'MEASURE':
            q_loc = line[1]
            c_loc = ''
            if line[2] != None:
                c_loc = line[2]

            self.__measure(q_loc, c_loc)

        # While instruction takes the form ('WHILE', test_loc, body)
        elif line[0] == 'WHILE':
            test_loc = line[1]
            body = line[2]
            self.__while(test_loc, body)

        # If instruction takes the form ('IF', c_reg_test_loc, if_branch, else_branch)
        elif line[0] == 'IF':
            if len(line) != 4:
                raise TypeError("Improper syntax in 'IF' statement.")
            # Gets the classical register test location
            test_loc = line[1]
            # This branch happens when no 'else' is specified
            if line[3] == None:
                then_branch = line[2]
                self.__if(test_loc, then_branch)

            # This branch only happens when an 'else' is specified
            else:
                then_branch = line[2]
                else_branch = line[3]
                self.__if(test_loc, then_branch, else_branch = else_branch)

        # Catches the empty instruction
        elif line[0] == None:
            pass

        else:
            raise sqerr.UnknownInstruction("Unable to parse '{}'.".format(str(line)))

    def __interpreter(self, program):
        '''
        Sequentially interprets each instruction in a program.
        '''

        if not isinstance(program, type(Program())):
            raise TypeError('Can only execute Program objects.')

        # If the program is empty, add a null instruction for execution purposes.
        if len(program) == 0:
            program.null()

        instruction_number = 0
        # Run through each line of the program
        for n, instruction in enumerate(program):
            instruction_number += 1
            try:
                self.__elementary_eval(instruction)
            except Exception as ex:
                # Generate error message
                message = ""
                message += type(ex).__name__
                message += " while parsing instruction {}: \n".format(instruction_number)
                message += program.format_instr(n)
                print(message)
                print(traceback.format_exc())
                sys.exit(1)

    def execute(self, program):
        '''
        Returns the contents of the classical register as a list after executing
        a program.
        '''
        self.__interpreter(program)

        output_c_reg = copy.deepcopy(self.__classical_reg)

        self.__reset()

        return output_c_reg

    def quantum_state(self, program):
        '''
        Returns the state of the quantum register as a Qubit object after
        executing a program. From a physical perspective, this is cheating!
        '''

        self.__interpreter(program)

        output_q_reg = Qubit(self.__quantum_reg.state())

        self.__reset()

        return output_q_reg

class Program():
    '''
    Provides a data structure for composing and organizing programs to run on
    qcVirtualMachine. Instructions of a program are stored internally in a list
    of tuples, where each tuple is a token corresponding to an instruction.
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

    def instructions(self):
        '''
        Helper method for returning a copy of the instructions.
        '''

        if len(self) == 0:
            return None

        else:
            return copy.deepcopy(self.__instructions)

    def add_instr(self, gate_target_tuple, kraus_ops=None, **params):
        '''
        Appends a quantum or classical instruction to self.__instructions. If
        a list of Kraus matricies is provided, the quantum instruction is
        generalized to a quantum operation. The list kraus_ops must have elements
        that are numpy ndarrays. If the gate is parametric, corresponding parameter
        values must be supplied as well. However, this isn't enforced by the
        Program class. An exception will only get raised at runtime on the
        qcVirtualMachine.
        '''

        # Check that the gate_target_tuple has the correct form.
        if not isinstance(gate_target_tuple, tuple):
            raise TypeError("gate_target_tuple must be a tuple.")

        if len(gate_target_tuple) < 2:
            raise TypeError("gate_target_tuple must have at least two elements (gate_name, target).")

        name = gate_target_tuple[0]

        if not isinstance(gate_target_tuple, tuple):
                raise TypeError('Argument must be a tuple of Gate name string followed by targets.')

        if not isinstance(name, str):
            raise TypeError('First element of argument must be a Gate name string.')

        for i in range(1, len(gate_target_tuple)):
            if not isinstance(gate_target_tuple[i], int) or gate_target_tuple[i] < 0:
                raise IndexError('Targets must be indexed as nonnegative integers.')

        # Check that the gate name isn't a keyword.
        if name in _keywords:
            raise NameError('{} is a protected keyword.'.format(name))

        # Check that the gate name isn't an empty string.
        if len(name) == 0:
            raise NameError('Gate name must be a nonempty string.')

        # Check that the gate name starts with a letter.
        if not name[0].isalpha():
            raise NameError('Gate name must start with a letter.')

        # Check that the gate name doesn't contain any protected chars.
        for char in name:
            if char in _protected_chars:
                raise NameError("Gate name contains protected char '{}'.".format(char))

        # Check that if kraus_ops are given, they are in the form of a list of
        # matrix like objects.
        if not isinstance(kraus_ops, type(None)):
            # Check that argument kraus_ops has the right form
            if not isinstance(kraus_ops, list):
                raise TypeError("kraus_ops must be a list of matricies.")
            if len(kraus_ops) < 2:
                raise TypeError("Must specify at least two Kraus operators for a quantum op.")

            # Check that each element of kraus_ops is a ndarray
            if not all(isinstance(op, type(np.array([]))) for op in kraus_ops):
                raise TypeError("Each operator in kraus_ops must be a numpy array.")

            # Check that each operator in kraus_ops has the correct shape.
            kraus_shape = kraus_ops[0].shape
            if kraus_shape[0] != kraus_shape[1]:
                raise sqerr.WrongShapeError("Kraus operators must be square matricies.")

            # Check that kraus_ops satisfy completeness
            identity = np.identity(kraus_shape[0])
            sum = np.matmul(np.conjugate(kraus_ops[0].T), kraus_ops[0])
            for i in range(1, len(kraus_ops)):
                sum += np.matmul(np.conjugate(kraus_ops[i].T), kraus_ops[i])

            if not np.allclose(sum, identity):
                raise sqerr.NormalizationError("Kraus operators must satisfy completeness relation.")

            for op in kraus_ops:
                if op.shape != kraus_shape:
                    raise sqerr.WrongShapeError("All Kraus operators must have same shape.")
                for row in op:
                    try:
                        len(row)
                    except:
                        raise TypeError("Rows of kraus_ops matricies must be list-like.")
                    for element in row:
                        try:
                            element + 5
                        except:
                            raise TypeError("Elements of kraus_ops matricies must be numeric.")

        if len(params) == 0:
            params = None

        self.__instructions.append((gate_target_tuple, params, kraus_ops))

    def rm_instr(self):
        if len(self.__instructions) > 0:
            self.__instructions.pop()
        else:
            pass

    def measure(self, qubit_loc, classical_loc=None):
        '''
        Adds a measurement instruction to the program.
        '''

        # If the qubit_loc isn't valid, throw an IndexError
        if not isinstance(qubit_loc, int) or qubit_loc < 0:
            raise IndexError('Qubit register location must be a nonnegative integer.')

        # If the classical_loc isn't valid, throw an IndexError
        if classical_loc != None:
            if not isinstance(classical_loc, int) or classical_loc < 0:
                raise IndexError('Classical register location must be a nonnegative integer.')

        self.__instructions.append(('MEASURE', qubit_loc, classical_loc))

    def gate_def(self, name, matrix_rep):
        '''
        Adds a 'GATEDEF' instruction to the program. If matrix_rep is not callable,
        then it has to be a matrix-like list or tuple, or a numpy ndarray. If
        matrix_rep is callable, then it has to be a function which returns a
        a matrix-like list or tuple, or a numpy ndarray when called with some
        number of **kwargs. If matrix_rep is callable but doesn't return a
        useable matrix representation of a quantum gate, an exception gets thrown
        at runtime on qcVirtualMachine.
        '''

        # Force name to string.
        name = str(name)

        # Check if name is empty.
        if len(name) == 0:
            raise NameError('Gate name string must be nonempty.')

        # Check if name is protected.
        if name in _keywords or name in gt.CLASSICAL_OPS or name in gt.STD_GATES:
            raise NameError('{} is a protected name.'.format(name))

        # Check that name starts with a letter.
        if not name[0].isalpha():
            raise NameError('Gate name must start with a letter.')

        # Check that no protected characters appear in name.
        for char in name:
            if char in _protected_chars:
                raise NameError("Gate name contains protected char '{}'.".format(char))

        # Check that matrix_rep is either a callable, a matrix-like combination
        # of lists and tuples, or a numpy array.
        if not callable(matrix_rep):
            if not isinstance(matrix_rep, list) or not isinstance(matrix_rep, tuple)\
                or not isinstance(matrix_rep, type(np.array([0]))):
                raise TypeError('matrix_rep must be callable, tuple, list, or a numpy array.')
            for row in matrix_rep:
                if not isinstance(row, (list, tuple, type(np.array([0])))):
                    raise TypeError('matrix_rep must be list-like of list-likes.')
                for element in row:
                    try:
                        element + 5
                    except:
                        raise TypeError('Elements of matrix_rep must be numeric.')


        self.__instructions.append(('GATEDEF', name, matrix_rep))

        # Now we return a convenience function to the user for generating an instruction
        # to apply the new gate.
        def gate_func(*target_qubits):
            return (name, *target_qubits)

        return gate_func

    def if_then_else(self, c_reg_test_loc, then_branch, else_branch=None):
        '''
        Adds the instruction to execute the subprogram 'then_branch' if the
        bit at classical register index 'c_reg_test_loc' is 1. Otherwise, if an 'else_branch'
        Program object is specified, that gets executed. If none is specified,
        execution passes on.
        '''

        # Check that the c_reg_test_loc is a nonnegative integer
        if not isinstance(c_reg_test_loc, int) or c_reg_test_loc < 0:
            raise IndexError("Classical register test index must be a nonnegative integer.")

        # Check that the branches are valid Program objects
        if not isinstance(then_branch, type(Program())):
            raise TypeError("Branches of 'IF' statement must be Program objects.")

        if else_branch != None:
            if not isinstance(else_branch, type(Program())):
                raise TypeError("Specified else branch of 'IF' statement must be a Program object.")

        ite_instruction = ('IF', c_reg_test_loc, then_branch, else_branch)

        self.__instructions.append(ite_instruction)

    def while_loop(self, c_reg_test_loc, loop_body):
        '''
        Adds the instruction to execute the subprogram 'loop_body' while the
        bit at classical register index 'c_reg_test_loc' is 1.
        '''

        # Check that the c_reg_test_loc is a nonnegative integer
        if not isinstance(c_reg_test_loc, int) or c_reg_test_loc < 0:
            raise IndexError("Classical register test index must be a nonnegative integer.")

        # Check that the body is a valid program object
        if not isinstance(loop_body, type(Program())):
            raise TypeError('Body of while statement must be a Program object.')

        loop_instruction = ('WHILE', c_reg_test_loc, loop_body)
        self.__instructions.append(loop_instruction)

    def null(self):
        null_instruction = (None, 0)
        self.__instructions.append(null_instruction)

    def format_instr(self, n):
        '''
        Formats the nth instruction in the program (zero-indexed) according to
        the SQUEAK language specification and then returns it as a string.
        '''

        if not isinstance(n, int) or n < 0:
            raise IndexError('Program instruction number must be a nonnegative integer.')

        instruction = self.__instructions[n]
        instr_rep = ""

        # Check for 'MEASURE' instruction ('MEASURE', qubit_loc, c_loc=None).
        if instruction[0] == 'MEASURE':
            instr_rep += instruction[0]
            instr_rep += ' '
            instr_rep += str(instruction[1])
            # Check if c_reg_loc is specified to save measurement.
            if instruction[2] != None:
                instr_rep += ' '
                instr_rep += str(instruction[2])

            instr_rep += ';\n'

        # Check for 'GATEDEF' instruction. (gdef, name, mrep)
        elif instruction[0] == 'GATEDEF':
            # First we construction the declaration line.
            instr_rep += instruction[0]
            # Append the gate name to the declaration line.
            instr_rep += ' '
            instr_rep += instruction[1]
            # If ParametricGate def, include parameter names as argument in printing.
            if callable(instruction[2]):
                # Get list of parameter names
                param_names = inspect.getfullargspec(instruction[2])[0]
                n_params = len(param_names)
                instr_rep += '('
                for i in range(n_params):
                    if i == n_params - 1:
                        instr_rep += param_names[i]
                        instr_rep += ')'
                    else:
                        instr_rep += param_names[i]
                        instr_rep += ', '

            instr_rep += ':'

            # For now just print the Python source code used to generate if
            # parametric gate.
            if callable(instruction[2]):
                instr_rep += '\n{\n'
                matrix_lines = inspect.getsourcelines(instruction[2])[0]
                for line in matrix_lines:
                    instr_rep += line
                instr_rep += '}\n'

            # Simpler if static gate.
            else:
                for row in instruction[2]:
                    instr_rep += '\n\t'
                    for element in row:
                        instr_rep += str(element)
                        instr_rep += ', '
                    # Remove trailing whitespace and comma
                    instr_rep = instr_rep[:-2]
                    instr_rep += ';'
                instr_rep += '\n'

        elif instruction[0] == 'IF':
            instr_rep += "IF("
            instr_rep += str(instruction[1]) + "):\n\t"

            # Generate THEN branch
            temp = instruction[2].__repr__()
            then_prog = ""
            for char in temp:
                if char == "\n":
                    then_prog += char + '\t'
                else:
                    then_prog += char
            instr_rep += "THEN(\n\t" + then_prog
            instr_rep += ") "

            # Generate ELSE branch
            if instruction[3] != None:
                temp = instruction[3].__repr__()
                else_prog = ""
                for char in temp:
                    if char == "\n":
                        else_prog += char + '\t'
                    else:
                        else_prog += char
            else:
                else_prog = ";\n\t"
            instr_rep += "ELSE(\n\t" + else_prog
            instr_rep += ")\n"

        elif instruction[0] == 'WHILE':
            instr_rep += "WHILE("
            instr_rep += str(instruction[1]) + "):\n\t"


            # Generate DO block
            temp = instruction[2].__repr__()
            do_block = ""
            for char in temp:
                if char == "\n":
                    do_block += char + '\t'
                else:
                    do_block += char
            instr_rep += "DO(\n\t" + do_block
            instr_rep +=")\n"

        elif instruction[0] == None:
            instr_rep += ";\n"

        # (Gate target tuple, params=None, kraus_ops=None)
        elif isinstance(instruction[0], tuple):
            gate_target_tuple = instruction[0]
            # If parameters are specified, include as args to gate name.
            if instruction[1] != None:
                instr_rep += gate_target_tuple[0] + '('
                for param_key in instruction[1]:
                    instr_rep += str(param_key) + '=' + str(instruction[1][param_key]) + ', '
                # Remove trailing whitespace and comma
                instr_rep = instr_rep[:-2] + ') '

            else:
                instr_rep += gate_target_tuple[0] + ' '

            # Now we print the targets.
            for i in range(1, len(gate_target_tuple)):
                if i == len(gate_target_tuple) - 1:
                    instr_rep += str(gate_target_tuple[i]) + ';'

                else:
                    instr_rep += str(gate_target_tuple[i]) + ' '

            # Remove trailing whitespace.
            instr_rep = instr_rep[:-1] + ';\n'

        else:
            raise ValueError("Unknown instruction encountered.")

        return instr_rep

    def __len__(self):
        return len(self.__instructions)

    def __repr__(self):
        program_rep = ""

        for i in range(len(self.__instructions)):
            instr_rep = ""
            instr_rep += self.format_instr(i)

            program_rep += instr_rep

        if len(self) == 0:
            program_rep = ';'

        return program_rep
