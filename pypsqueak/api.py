import numpy as np
import copy
import cmath

from pypsqueak.squeakcore import Qubit, Gate
import pypsqueak.errors as sqerr

class qReg:
    '''
    A high-level primitive which provides users with a means of interfacing with
    a quantum device (simulated in this implementation).
    '''

    def __init__(self, n_qubits = 1):
        if n_qubits < 1:
            raise ValueError("A qReg must have length of at least 1.")

        init_state = [0 for i in range(2**n_qubits)]
        init_state[0] = 1
        self.__q_reg = Qubit(init_state)
        self.__killed = False

    def measure(self, target):
        '''
        Performs a projective measurement on the qubit at the address ``target``.
        1) Determine measurement outcomes
        2) Compute probability of each measurement
        3) Use probabilities to randomly pick a result
        4) Project onto the corresponding eigenspace (build projector out of
           sum of corresponding eigenvectors |e_i><e_i| represented in comp basis)
        '''

        if self.__killed:
            raise sqerr.IllegalRegisterReference('Measurement attempted on dereferenced register.')

        if not isinstance(target, int) or target < 0:
            raise TypeError('Quantum register address must be nonnegative integer.')

        if target > len(self) - 1:
            raise ValueError('Specified quantum register address out of range.')

        # We use the relative amplitudes |0> or |1> measurements to generate
        # corresponding probability weights.
        amplitudes_for_zero = []
        amplitudes_for_one = []

        # Decompose the state into a dict of basis label and amplitude pairs.
        basis_states = self.__q_reg.computational_decomp()
        for state_label in basis_states:
            if int(state_label[-1 - target]) == 0:
                amplitudes_for_zero.append(basis_states[state_label])

            if int(state_label[-1 - target]) == 1:
                amplitudes_for_one.append(basis_states[state_label])

        # We then use the sorted amplitudes to generate the probability weights
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

        # Next we project the state of q_reg onto the eigenbasis corresponding
        # to the measurement result.
        projector_diag = []
        # If the qubit at address target in state_label == measurement, it is
        # a part of the eigenbasis.
        for state_label in basis_states:
            if int(state_label[-1 - target]) == measurement:
                projector_diag.append(1)

            else:
                projector_diag.append(0)

        projector_operator = np.diag(projector_diag)
        new_state = np.dot(projector_operator, self.__q_reg.state())
        # Note that the change_state() method automatically normalizes new_state
        self.__q_reg.change_state(new_state)

        return measurement

    def measure_observable(self, observable):
        '''
        Performs a projective measurement on the qubit at the address ``target``.
        1) Determine measurement outcomes
        2) Compute probability of each measurement
        3) Use probabilities to randomly pick a result
        4) Project onto the corresponding eigenspace (build projector out of
           sum of corresponding eigenvectors |e_i><e_i| represented in comp basis)

        Default behavior is to append identities to operator if smaller than the
        qReg i.e. for qReg(3), X(x)X gets promoted to X(x)X(x)I.
        '''

        if not isinstance(observable, type(qOp())):
            raise TypeError("Argument of measure_observable() must be a qOp.")

        if len(self) < observable.size():
            raise sqerr.WrongShapeError("Observable larger than qReg.")

        if len(self) > observable.size():
            iden = qOp()
            for i in range(len(self) - observable.size()):
                observable = observable.kron(iden)

        # Determine normalized eigenvalue/vector pairs.
        e_vals, e_vecs = np.linalg.eig(observable._qOp__state.state())

        # print("Vectors:", e_vecs)
        # print("Evals:", e_vals)

        # Compute probabilities
        probabilities = []
        current_state = self.__q_reg.state()
        for e_vec in e_vecs:
            amplitude = np.dot(current_state, e_vec)
            probabilities.append(amplitude * amplitude.conj())

        # print("Probs:", probabilities)

        # Choose measurement result
        measurement_index = np.random.choice([i for i in range(len(e_vals))], p=probabilities)
        measurement_result = e_vals[measurement_index]

        # print("Result:", measurement_result)

        # Build subspace corresponding to eigenvalue
        subspace_basis = []
        for i in range(len(e_vals)):
            if e_vals[i] == measurement_result:
                subspace_basis.append(e_vecs[:,i])

        # print("Subspace basis:", subspace_basis)

        # Make projection operator
        projector = np.outer(subspace_basis[0], subspace_basis[0])
        for i in range(1, len(subspace_basis)):
            projector += np.outer(subspace_basis[i], subspace_basis[i])

        # print("Projection operator:", projector)

        new_state = np.dot(projector, current_state)
        self.__q_reg.change_state(new_state)

        return measurement_result

    def peek(self):
        '''
        Returns a description of the simulated qReg's state as a ket. Note that this
        is impossible on hardware implementations as a consequence of the
        no-cloning theorem.
        '''

        if self.__killed:
            raise sqerr.IllegalRegisterReference('Dereferenced register encountered.')

        return str(self.__q_reg)

    def dump_state(self):
        '''
        Returns a copy of the quantum register's state vector. Note that this
        is impossible on hardware implementations as a consequence of the
        no-cloning theorem.
        '''

        if self.__killed:
            raise sqerr.IllegalRegisterReference('Dereferenced register encountered.')

        return self.__q_reg.state()

    def __iadd__(self, n_new_qubits):
        '''
        Adds ``n_new_qubits`` qubits to the register in the |0> state.
        '''

        if self.__killed:
            raise sqerr.IllegalRegisterReference('Attempt to add Qubits to dereferenced register.')

        if not isinstance(n_new_qubits, int) or n_new_qubits < 0:
            raise ValueError("Can only add a positive integer number of qubits to quantumRegister.")

        new_register = Qubit()
        for i in range(n_new_qubits - 1):
            new_register = new_register.qubit_product(Qubit())

        self.__q_reg = new_register.qubit_product(self.__q_reg)

        return self

    # def __isub__(self, n):
    #     '''
    #     Drops ``n`` qubits from the end of the register via projection. If
    #     the register initially has ``L`` qubits, then the result of the projection
    #     will be to leave the register in the subspace spanned by the first ``m = L - n``
    #     qubits. More precisely, basis states |x_0>|y> and |x_1>|y> are identified
    #     with each other and components are added together in the new basis state |y>.
    #     The resulting ``2**m`` dimension vector is then normalized.
    #     '''
    #
    #     if self.__killed:
    #         raise sqerr.IllegalRegisterReference('Qubit removal attempted on dereferenced register.')
    #
    #     if not isinstance(n, int) or n < 0:
    #         raise ValueError("Can only remove a negative integer number of qubits from quantumRegister.")
    #
    #     m = len(self) - n
    #     if m < 1:
    #         raise ValueError("Resulting quantum register has length <= 0.")
    #
    #     projector_diag = [1 for i in range(2**m)] + [0 for i in range(2**len(self) - 2**m)]
    #     projector = np.diag(projector_diag)
    #     old_register = self.__q_reg.state()
    #     new_register = []
    #
    #     for i in range(2**m):
    #         projected_component = 0
    #         for j in range(2**n):
    #             current_index = i + j*2**m
    #             projected_component += old_register[current_index]
    #         new_register.append(projected_component)
    #
    #     self.__q_reg.change_state(new_register)
    #
    #     return self

    def __imul__(self, some_reg):
        '''
        Concatentates the register with some_reg (|a_reg> *= |some_reg> stores
        |a_reg>|some_reg> into ``a_reg``).
        '''
        if self.__killed:
            raise sqerr.IllegalRegisterReference('Concatentation attempted on dereferenced register.')

        if not isinstance(some_reg, type(qReg())):
            raise ValueError("Cannot concatentate a qReg to a non-qReg.")

        self.__q_reg = self.__q_reg.qubit_product(some_reg._qReg__q_reg)

        some_reg._qReg__killed = True

        return self

    def __mul__(self, another_reg):
        '''
        For concatentating the register with another_reg
        (|new> = |reg> * |another_reg> stores the product into ``new``).
        '''

        if self.__killed:
            raise sqerr.IllegalRegisterReference('Concatentation attempted on dereferenced register.')

        new_register = qReg()
        new_state = self.__q_reg.qubit_product(another_reg._qReg__q_reg)

        self.__killed = True
        another_reg._qReg__killed = True

        new_register._qReg__q_reg.change_state(new_state.state())

        return new_register

    def __len__(self):
        if self.__killed:
            raise sqerr.IllegalRegisterReference('Dereferenced register encountered.')

        return len(self.__q_reg)

    def __copy__(self):
        raise sqerr.IllegalCopyAttempt('Cannot copy a qReg.')

    def __deepcopy__(self, memo):
        raise sqerr.IllegalCopyAttempt('Cannot copy a qReg.')

    def __repr__(self):
        if not self.__killed:
            return "qReg({})".format(len(self))
        else:
            return "Dereferenced qReg"


# Parametric qOps can be accommodated by defining a function that returns
# a qOp.
class qOp:

    '''
    A high-level primitive for representing unitary gates. In this implementation,
    noise can be simulated by instantiating a ``qOp`` with the kwarg ``kraus_ops``,
    a list of operation elements characterizing a noisy quantum operation.
    '''

    def __init__(self, matrix_rep=[[1, 0], [0, 1]], kraus_ops=None):
        self.__state = Gate(matrix_rep)
        self.__noise_model = kraus_ops

    def set_noise_model(self, kraus_ops):
        '''
        Changes the noise model on the qOp to that specified by ``kraus_ops``.
        If ``kraus_ops = None``, then the qOp is noiselessly emulated. Note that
        this method would be absent from a hardware implementation of SQUEAK.
        '''

        self.__noise_model = kraus_ops

    def size(self):
        return len(self.__state)

    def shape(self):
        return (2**len(self.__state),) * 2

    def on(self, q_reg, *targets):
        '''
        Applies the operation to a quantum register.
        '''

        if q_reg._qReg__killed:
            raise IllegalRegisterReference("Cannot operate on a dereferenced register.")

        # Check that at least one quantum_reg location is specified when gate and register
        # sizes don't match.
        if len(targets) == 0 and self.size() != len(q_reg):
            raise IndexError('One or more targets must be specified for gate and register of different size.')

        ### OLD SWAP METHOD ###
        # Or, make dummy targets when gate and register sizes do match. These dummy targets
        # simply preserve the order of the qubits in the register.
        if len(targets) == 0 and self.size() == len(q_reg):
            targets = [i for i in range(self.size())]
        ### OLD SWAP METHOD END ###

        # Check that there are no duplicate register locations for the instruction
        if len(targets) != len(set(targets)):
            raise ValueError('Specified quantum register targets must be unique.')

        # Check that the gate size matches the number of quantum_reg locations
        if self.size() != len(targets):
            raise sqerr.WrongShapeError('Number of registers must match number of qubits gate operates on.')

        # Check that all the register locations are nonnegative integers
        for address in targets:
            if not isinstance(address, int):
                raise TypeError('Quantum register addresses must be integer.')

            if address < 0:
                raise ValueError('Quantum register addresses must be nonnegative.')

        # Now we run some more checks on the validity of the Kraus operators
        if self.__noise_model != None:

            # Check that the size of the Kraus operators agrees with the gate size
            kraus_shape = self.__noise_model[0].shape
            gate_shape = self.__state.state().shape
            if gate_shape != kraus_shape:
                raise sqerr.WrongShapeError("Size mismatch between Kraus operators and gate.")

        # If any of the specified quantum_reg addresses have not yet been initialized,
        # initialize them (as well as intermediate reg locs) in the |0> state
        if max(targets) > len(q_reg) - 1:
            q_reg += max(targets) - len(q_reg) + 1

        # Initialize an identity gate for later use.
        iden = Gate()

        # If no targets are specified, qOp acts on unpermutated state.
        if len(targets) == 0:
            swap = np.eye(2**len(q_reg))
            swap_inverse = swap
        else:
            swap, swap_inverse = self.__generate_swap(q_reg, *targets)

        before_swap_and_op = q_reg._qReg__q_reg.state()
        after_swap_before_op = np.dot(swap, before_swap_and_op)

        ### OLD SWAP METHOD ###
        # # Swap operational qubits into order specified by targets args.
        # # First we generate pairs of swaps to make, using a set to avoid duplicates.
        # # q_flag increases from initially pointing at the zero qubit by one on each
        # # loop.
        # q_flag = 0
        # swap_pairs = set()
        # for reg_loc in targets:
        #     swap_pairs.add((q_flag, reg_loc))
        #     q_flag += 1
        #
        # # Remove duplicates
        # swap_pairs = set((a,b) if a<=b else (b,a) for a,b in swap_pairs)
        #
        # # Perform swaps of qubits into the operational order (first specified qubit
        # # in q_reg_loc 0, second in q_reg_loc 1, etc.)
        #
        # for swap_pair in swap_pairs:
        #     self.__swap(q_reg, swap_pair[0], swap_pair[1])
        ### OLD SWAP METHOD END ###

        # If the gate and size of the quantum register match, just operate with the gate
        if len(q_reg) == self.size():
            operator = self.__state

        # If the register size is larger, we need to raise the gate (I^n tensored with gate,
        # since operational order means the target qubits are ordered into the lowest
        # register adresses by this point).
        elif len(q_reg) > self.size():
            left_eye = iden
            for i in range(len(q_reg) - self.size() - 1):
                left_eye = left_eye.gate_product(iden)

            operator = left_eye.gate_product(self.__state)

        # If no Kraus operators are specified, evaluation of new register state is trivial
        if self.__noise_model == None:
            # print("\tOp:", operator.state())
            # print("\tState:", q_reg.peek())
            after_swap_after_op = np.dot(operator.state(), after_swap_before_op)
            # q_reg._qReg__q_reg.change_state(after_swap_after_op)
            # print("\tNew State:", q_reg.peek())

        else:
            # We randomly choose one of the Kraus operators to apply, then we
            # generate a corresponding gate to hand over to the __instr() method
            current_state = np.dot(operator.state(), after_swap_before_op)
            probs = []
            new_state_ensemble = []

            # Generate an ensemble of states transformed according to Kraus ops in the
            # form of a list of transformed state vector, and a corresponding
            # list of probability weights for each tranformation
            for op in self.__noise_model:
                # Raise each operator if necessary
                if len(q_reg) > np.log2(op.shape[0]):
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
            after_swap_after_op = new_state_ensemble[new_state_index]
            # q_reg._qReg__q_reg.change_state(after_swap_after_op)

        new_reg_state = np.dot(swap_inverse, after_swap_after_op)
        q_reg._qReg__q_reg.change_state(new_reg_state)
        ### OLD SWAP METHOD ###
        # # Swap qubits back into original order
        # for swap_pair in swap_pairs:
        #     self.__swap(q_reg, swap_pair[0], swap_pair[1])
        ### OLD SWAP METHOD END ###

    def __generate_swap(self, q_reg, *targets):
        '''
        Given a list of targets, generates matrix (and inverse) to swap targets
        into lowest qubit slot in register. Remaining qubits in register get
        bumped up, perserving order.

        Example: |abcdef> with targets = [3, 0, 4, 1] goes to |adebfc>
        '''

        # First generate list of sorted qubit indicies.
        new_order = []
        for target in targets:
            new_order.append(target)

        for i in range(len(q_reg)):
            if not i in new_order:
                new_order.append(i)

        # Use new_order to generate a corresponding permutation matrix.
        perm_matrix = np.zeros((len(new_order), len(new_order)))
        for i in range(len(q_reg)):
            perm_matrix[i][new_order[i]] = 1

        swap_matrix = np.zeros((2**len(q_reg), 2**len(q_reg)))
        # Iterate through each basis label, applying permutation matrix to generate
        # new labels. Then, Convert old and new labels to ints. The unitary matrix
        # implementing the desired qubit swap is given by U[new][old] = 1,
        # for each transformed pair and has 0s everywhere else.
        for basis_label in q_reg._qReg__q_reg.computational_decomp():
            old_label_vector = []
            for ch in basis_label[::-1]:
                old_label_vector.append(int(ch))
            new_label_vector = np.dot(perm_matrix, list(old_label_vector))
            new_label = ""
            for ch in new_label_vector:
                new_label += str(int(ch))
            new_label = new_label[::-1]
            row = int(new_label, 2)
            col = int(basis_label, 2)
            swap_matrix[row][col] = 1
        swap_matrix_inverse = swap_matrix.T

        # Check that the transpose is the inverse.
        if not np.array_equal(np.dot(swap_matrix, swap_matrix_inverse), np.eye(2**len(q_reg))):
            raise ValueError("Nonunitary swap encountered.")

        return swap_matrix, swap_matrix_inverse

    ### OLD SWAP METHOD ###
    # def __swap(self, q_reg, i, j):
    #     # Method for swapping the ith and jth qubits in a quantum register.
    #     # Swaps higher of i and j by taking the upper index down abs(i - j) times,
    #     # and then taking the lower index up abs(i - j) - 1 times
    #     if not isinstance(i, int) or not isinstance(j, int):
    #         raise IndexError('Quantum register indicies must be integer-valued.')
    #
    #     if i == j:
    #         return None
    #
    #     upper_index = max(i, j)
    #     lower_index = min(i, j)
    #     difference = abs(i - j)
    #
    #     # Bring upper down with 'difference' elementary swaps (left prepends SWAP, right post)
    #     for k in range(difference):
    #         # Indicies for elementary swap on kth loop
    #         simple_lower = upper_index - (1 + k)
    #         simple_upper = upper_index - k
    #
    #         self.__elementary_swap(q_reg, simple_lower, simple_upper)
    #
    #     # Bring lower up with (difference - 1) elementary swaps (left prepends SWAP, right post)
    #     for k in range(difference - 1):
    #         # Indicies for elementary swap on kth loop
    #         simple_lower = lower_index + (1 + k)
    #         simple_upper = lower_index + (2 + k)
    #
    #         self.__elementary_swap(q_reg, simple_lower, simple_upper)
    #
    # def __elementary_swap(self, q_reg, simple_lower, simple_upper):
    #     # Helper method to swap adjacent qubits in the quantum register.
    #
    #     # Raise IndexError if swap indicies reference location in the quantum register
    #     # that doesn't exist
    #     if simple_lower < 0 or simple_lower > (len(q_reg) - 1):
    #         raise IndexError("One or more register locations specified in swap doesn't exist.")
    #
    #     if simple_upper < 0 or simple_upper > (len(q_reg) - 1):
    #         raise IndexError("One or more register locations specified in swap doesn't exist.")
    #
    #     # Initialize identity and swap gates for later use (and throw away target qubit)
    #     iden = Gate()
    #     swap = Gate([[1, 0, 0, 0],
    #                  [0, 0, 1, 0],
    #                  [0, 1, 0, 0],
    #                  [0, 0, 0, 1]])
    #
    #     # Note that lower index corresponds to right-hand factors, upper index to left-hand
    #     number_right_eye = int(simple_lower)
    #     number_left_eye = int(len(q_reg) - simple_upper - 1)
    #
    #     if number_left_eye > 0 and number_right_eye > 0:
    #         # Prep identity factors
    #         left_eye = iden.gate_product(*[iden for l in range(number_left_eye - 1)])
    #         right_eye = iden.gate_product(*[iden for l in range(number_right_eye - 1)])
    #
    #         raised_swap = left_eye.gate_product(swap, right_eye)
    #         new_swap_state = np.dot(raised_swap.state(), q_reg._qReg__q_reg.state())
    #         q_reg._qReg__q_reg.change_state(new_swap_state)
    #
    #     elif number_left_eye > 0 and number_right_eye == 0:
    #         # Prep identity factors
    #         left_eye = iden.gate_product(*[iden for l in range(number_left_eye - 1)])
    #
    #         raised_swap = left_eye.gate_product(swap)
    #         new_swap_state = np.dot(raised_swap.state(), q_reg._qReg__q_reg.state())
    #         q_reg._qReg__q_reg.change_state(new_swap_state)
    #
    #     elif number_left_eye == 0 and number_right_eye > 0:
    #         # Prep identity factors
    #         right_eye = iden.gate_product(*[iden for l in range(number_right_eye - 1)])
    #
    #         raised_swap = swap.gate_product(right_eye)
    #         new_swap_state = np.dot(raised_swap.state(), q_reg._qReg__q_reg.state())
    #         q_reg._qReg__q_reg.change_state(new_swap_state)
    #
    #     elif number_left_eye == 0 and number_right_eye == 0:
    #         raised_swap = swap
    #         new_swap_state = np.dot(raised_swap.state(), q_reg._qReg__q_reg.state())
    #         q_reg._qReg__q_reg.change_state(new_swap_state)
    ### OLD SWAP METHOD END ###

    def dagger(self):
        '''
        Returns the Hermitian conjugate.
        '''

        herm_trans = self.__state.state().conj().T

        return qOp(herm_trans)

    def __mul__(self, another_op):
        '''
        Returns the matrix product (another_op)(some_op) i.e. with another_op
        acting second. The resulting noise model is that of ``some_op``.
        '''
        if self.size() != another_op.size():
            raise sqerr.WrongShapeError("qOp size mismatch.")

        product = np.dot(another_op._qOp__state.state(), self.__state.state())

        return qOp(product, kraus_ops=self.__noise_model)

    def kron(self, another_op, *more_ops):
        '''
        Returns the tensor product (Kronecker product) self (x) another_op.
        Optionally computes self (x) another_op (x) *more_ops for more qOps.
        '''

        if not isinstance(another_op, type(qOp())):
            raise TypeError("Arguments must be qOp objects.")

        matrix_reps = [another_op._qOp__state]
        for op in more_ops:
            if not isinstance(op, type(qOp())):
                raise TypeError("Arguments must be qOp objects.")
            matrix_reps.append(op._qOp__state)
        result_matrix = self.__state.gate_product(*matrix_reps).state()

        return qOp(result_matrix)

    def __repr__(self):

        return str(self.__state)

class qOracle(qOp):
    '''
    Implements a transformation corresponding to bitwise XOR with classical
    function f: U_f|x>|y> = |x>|y XOR f(x)>. Note XOR reduces to mod 2 addition
    when y and f(x) are both one bit long.
    '''

    def __init__(self, func, n, m=1, kraus_ops=None):
        if not isinstance(n, int) or not isinstance(m, int):
            raise TypeError('Dimension exponents n and m must be integer.')
        if n < 1 or m < 1:
            raise ValueError('Dimension exponents n and m must be positive.')
        if not callable(func):
            raise TypeError('First argument of qOracle must be callable.')

        self.__classical_func = func
        self.__domain_exp = n
        self.__range_exp = m

        # Check that the function values are valid.
        for value in [func(i) for i in range(2**n)]:
            if not isinstance(value, int):
                raise TypeError('Range of input function contains non-integers.')
            if value < 0 or value > 2**m - 1:
                raise ValueError('Range of input function out of bounds.')

        super().__init__(self.__generate_matrix_rep(), kraus_ops=kraus_ops)

    def classical_func(self, x_val):
        '''
        Returns the classical value of the function implemented by the ``qOracle``.
        '''

        if not isinstance(x_val, int):
            raise TypeError("Classical function maps ints to ints.")

        if x_val < 0 or x_val > 2**self.__domain_exp - 1:
            raise ValueError("Classical function input out of bounds.")

        return self.__classical_func(x_val)

    def __generate_matrix_rep(self):
        '''
        Generates the oracle for the register in state |x>|y>.
        '''

        dim = self.__range_exp + self.__domain_exp
        matrix_rep = np.zeros((2**dim, 2**dim))
        f_vals = [i for i in range(2**self.__range_exp)]
        col = 0
        for x in range(2**self.__domain_exp):
            for y in range(2**self.__range_exp):
                row = "{0:b}".format(x)
                row += "{0:b}".format(y ^ self.classical_func(x))
                row = int(row, 2)
                matrix_rep[row][col] += 1
                col += 1

        return matrix_rep

    def __repr__(self):
        return "qOracle({}, {})".format(self.__domain_exp, self.__range_exp)
