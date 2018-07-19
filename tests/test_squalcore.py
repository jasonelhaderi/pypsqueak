# Standard modules
import unittest
import numpy as np
import cmath

# Modules to test (context simply enables importing the 'squal' package)
import context
import squal.gates as gt
from squal.squalcore import Qubit, Gate
import squal.api as sq
import squal.errors as sqerr

# Qubit Tests

class QubitValidInput(unittest.TestCase):
    valid_qubits = [[1.0, 0],
                    [0.0, 1],
                    [1.0j, 0],
                    [1.0j/np.sqrt(2), -1j/np.sqrt(2)],
                    [24.0, 6],
                    (24.0, 6),
                    [1 - 1j, 0j],
                    [-25.0, 9j],
                    [-25.0, 9j, 3, 5]]

    def setUp(self):
        self.test_qubit = sq.Qubit()

    def test_normalize_valid_input(self):
        '''self.__normalize() should rescale valid new states to be unit vectors'''

        for vec in self.valid_qubits:
            # get machine epsilon for relative error
            mach_eps = np.finfo(type(vec[0])).eps

            # change state and compute inner product
            self.test_qubit.change_state(vec)
            state_dual = np.conjugate(self.test_qubit.state())

            # if inner product != 1, then test_qubit.normalize() failed
            inner_product = np.dot(self.test_qubit.state(), state_dual)

            # cmath is used so we don't discard errors in the imaginary part
            norm_query = cmath.isclose(1, inner_product, rel_tol=10*mach_eps)
            self.assertTrue(norm_query)

    def test_known_qubit_product(self):
        '''Verifies proper result for self.qubit_product()'''

        product_state_01 = np.array([0, 1, 0, 0])
        q1 = sq.Qubit([0, 1])
        test_product_state_01 = self.test_qubit.qubit_product(q1)

        np.testing.assert_array_equal(product_state_01, test_product_state_01.state())

class QubitInvalidInput(unittest.TestCase):

    def setUp(self):
        self.test_qubit = sq.Qubit()

    def test_not_list_or_tuple(self):
        '''self.__validate_state.() should fail for self.change_state(non-list or -tuple)'''

        not_list_or_tuple = [{'key':'value'},
                             4,
                             'string']
        for candidate in not_list_or_tuple:
            self.assertRaises(TypeError, self.test_qubit.change_state, candidate)

    def test_non_numeric_elements(self):
        '''self.__validate_state.() should fail for self.change_state(list with non-numeric elements)'''

        non_numeric_elements = [['string'],
                                [['string'], ['string']],
                                [[34, 25]],
                                [[24], [25]],
                                ([24], (25)),
                                [[24, 35], [-2]],
                                [[-1, 3], [4, 17], 2],
                                [{'key':'value'}, {'key':'value'}],
                                [(24, 35), (-2)]]
        for candidate in non_numeric_elements:
            self.assertRaises(TypeError, self.test_qubit.change_state, candidate)

    def test_wrong_length(self):
        '''self.__validate_state.() should fail for self.change_state(input with length not even)'''

        wrong_length = [[0, 1, 34],
                        [7]]
        for candidate in wrong_length:
            self.assertRaises(sqerr.WrongShapeError, self.test_qubit.change_state, candidate)

    def test_zero_vector(self):
        '''self.__validate_state.() should fail for self.change_state(null vector, list, or tuple)'''

        null_vectors = [[0, 0],
                        [],
                        ()]
        for null_vec in null_vectors:
            self.assertRaises(sqerr.NullVectorError, self.test_qubit.change_state, null_vec)

    def test_empty_input_qubit_product(self):
        '''qubit_product with empty argument should raise a TypeError'''

        self.assertRaises(TypeError, sq.Qubit().qubit_product)

# Gate tests

class GateValidInput(unittest.TestCase):

    X = np.array([[0, 1],
                 [1, 0]])
    Y = np.array([[0, -1j],
                 [1j, 0]])
    Z = np.array([[1, 0],
                 [0, -1]])
    I = np.array([[1, 0],
                 [0, 1]])
    valid_gates = [X, Y, Z, I]

    def setUp(self):
        self.gate_x = sq.Gate(self.X.tolist())
        self.gate_y = sq.Gate(self.Y.tolist())
        self.gate_z = sq.Gate(self.Z.tolist())
        self.gate_i = sq.Gate()

    def test_pauli_gates(self):
        '''Basic Pauli matricies should initialize correctly'''

        test_gates = [self.gate_x, self.gate_y, self.gate_z, self.gate_i]
        test_gates = zip(self.valid_gates, test_gates)

        for gate in test_gates:
            np.testing.assert_array_equal(gate[0], gate[1].state())

class GateInvalidInput(unittest.TestCase):

    def test_non_list_non_tuple(self):
        '''Gate Should fail to initialize if input not list-like of rows'''

        not_list_of_rows = [2,
                            'string',
                            ['some', 'list'],
                            None]

        for candidate in not_list_of_rows:
            self.assertRaises(ValueError, sq.Gate, candidate)

    def test_not_square(self):
        '''Gate should fail to initialize if input doesn't have shape (n, n)'''

        wrong_shapes = [[[]],
                        [[1, 2]],
                        [[1, 2, 3],
                         [1, 3, 4]],
                        [[0, 0],
                         [0, 0, 1]],
                         ([1, 2],
                          (1, 2, 3))]
        for candidate in wrong_shapes:
            self.assertRaises(sqerr.WrongShapeError, sq.Gate, candidate)

    def test_not_even(self):
        '''Gate should fail if matrix is not 2nX2n for integer n'''

        uneven_shape = [[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]

        self.assertRaises(sqerr.WrongShapeError, sq.Gate, uneven_shape)

    def test_not_unitary(self):
        '''Gate should fail to initialize non-unitary matrix'''

        M1 = [[1, 5],
              [10, 7]]
        M2 = [[0, 1j],
              [1j + 17, 0]]
        M3 = [[4, 0],
              [0, -3]]
        M4 = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

        non_unitary_matricies = [M1, M2, M3, M4]
        for matrix in non_unitary_matricies:
            self.assertRaises(sqerr.NonUnitaryInputError, sq.Gate, matrix)

class GateProductValidInput(unittest.TestCase):

    def test_known_two_qubit_gates(self):
        '''Checks that known two-qubit gates are formed by gate_product with one arg'''

        gate_i = sq.Gate()
        gate_z = sq.Gate([[1, 0],
                          [0, -1]])
        gate_i_prod_i = gate_i.gate_product(gate_i)
        gate_i_prod_z = gate_i.gate_product(gate_z)
        gate_i_squared = sq.Gate([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        gate_i_times_z = sq.Gate([[1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, -1]])
        prod_value_pairs = [(gate_i_prod_i.state(), gate_i_squared.state()),
                            (gate_i_prod_z.state(), gate_i_times_z.state())]

        for test, result in prod_value_pairs:
            np.testing.assert_array_equal(test, result)

    def test_empty_product(self):
        '''gate_product should return self with no arguments'''

        gate_z = sq.Gate([[1, 0],
                          [0, -1]])
        gate_empty_arg = gate_z.gate_product()

        np.testing.assert_array_equal(gate_z.state(), gate_empty_arg.state())

    def test_two_args_identity(self):
        '''Checks that the tensor product of two identities with identity works'''

        gate_i = sq.Gate()
        gate_i_cubed = gate_i.gate_product(gate_i, gate_i)
        gate_should_equal = sq.Gate(np.eye(8).tolist())

        np.testing.assert_array_equal(gate_i_cubed.state(), gate_should_equal.state())

class GateProductInvalidInput(unittest.TestCase):

    def test_non_gate_input(self):
        '''gate_product should fail with non Gate() input'''

        not_gates = [2, 16, [1, 2], [[1, 2], [2, 3]], []]

        for candidate in not_gates:
            self.assertRaises(TypeError, sq.Gate().gate_product, candidate)

#QCSim tests

class QCSimSuccess(unittest.TestCase):

    def setUp(self):
        # Test machine
        self.test_qc = sq.QCSim()

        # Test program
        self.test_program = sq.Program()

    def test_known_measurement_results(self):
        '''Verifies that the proper post-measurement state occurs in several cases'''

        # QCSim is initialized in the |0> state, so first let's measure a freshly
        # initialized QCSim, and store what should be zero in the 1st classical register
        # location
        p = sq.Program()
        p.measure(0, 1)
        q_reg_output, c_reg_output = self.test_qc.execute(p)
        np.testing.assert_array_equal(q_reg_output, np.array([1, 0]))
        np.testing.assert_array_equal(c_reg_output, [0, 0])

        # Now let's remove that instruction from the program and see that the X
        # gate gives a |1> state by saving the measurement into the 6th classical
        # register location
        p.rm_instr()
        p.add_instr(gt.X(0))
        p.measure(0, 6)
        q_reg_output, c_reg_output = self.test_qc.execute(p)
        np.testing.assert_array_equal(q_reg_output, np.array([0, 1]))
        np.testing.assert_array_equal(c_reg_output, [0, 0, 0, 0, 0, 0, 1])

        # Now let's reset the program to initialize the state |100>, and then measure
        # the 1st and 2nd qubits into classical register locations 3 and 2, respectively,
        # and then measure the 0th qubit without storing the result
        while len(p) > 0:
            p.rm_instr()

        p.add_instr(gt.X(2))
        p.measure(1, 3)
        p.measure(2, 2)
        p.measure(0)
        q_reg_output, c_reg_output = self.test_qc.execute(p)
        np.testing.assert_array_equal(q_reg_output,\
                                      np.array([0, 0, 0, 0, 1, 0, 0, 0]))
        np.testing.assert_array_equal(c_reg_output, [0, 0, 1, 0])

    def test_known_instr_results(self):
        '''
        Verifies the output of several known instructions.
        '''

        test_programs = [sq.Program() for i in range(3)]

        # Takes |0> to |1>
        test_programs[0].add_instr(gt.X(0))

        # Takes |0> to |0001>
        test_programs[1].add_instr(gt.X(0))
        test_programs[1].add_instr(gt.I(3))

        # Takes |0> to (1/sqrt(2))(|000> - |100>)
        test_programs[2].add_instr(gt.I(2))
        test_programs[2].add_instr(gt.X(2))
        test_programs[2].add_instr(gt.H(2))

        results = [np.array([0, 1]),
                   np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                   1/np.sqrt(2) * np.array([1, 0, 0, 0, -1, 0, 0, 0])]

        for test_pair in zip(test_programs, results):
            q_reg_output = self.test_qc.execute(test_pair[0])[0]
            np.testing.assert_array_almost_equal(q_reg_output, test_pair[1])

    def test_add_distant_qubit(self):
        '''
        The private self.__instr() method called with for a non-extant target qubit
        should initialize filler qubits in the |0> state.
        '''

        i_gate = gt.I(0)[0]
        self.test_qc._QCSim__instr(i_gate, 2)
        state_000 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(self.test_qc.quantum_reg(), state_000)

    def test_known_swaps(self):
        '''
        Verifies known swaps in the private self.__swap() method.
        '''
        # We use a hard-coded identity gate to initialize extra qubits
        i_gate = gt.I(0)[0]

        # Verify that |001> gets swapped to |100>
        self.test_qc._QCSim__instr(i_gate, 2)
        self.test_qc._QCSim__quantum_reg.change_state([0, 1, 0, 0, 0, 0, 0, 0])
        self.test_qc._QCSim__swap(0, 2)
        state_100 = np.array([0, 0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_array_almost_equal(self.test_qc.quantum_reg(), state_100)

        self.test_qc._QCSim__reset()

        # Verify that |100> gets swapped to |010> when qubits 1 and 2 are swapped
        self.test_qc._QCSim__instr(i_gate, 2)
        self.test_qc._QCSim__quantum_reg.change_state([0, 0, 0, 0, 1, 0, 0, 0])
        self.test_qc._QCSim__swap(1, 2)
        state_010 = np.array([0, 0, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(self.test_qc.quantum_reg(), state_010)

        self.test_qc._QCSim__reset()

        # Verify that (|011> - |010>)/sqrt(2) gets swapped to (|101> - |100>)/sqrt(2)
        # when qubits 1 and 2 are swapped
        self.test_qc._QCSim__instr(i_gate, 2)
        self.test_qc._QCSim__quantum_reg.change_state([0, 0, -1, 1, 0, 0, 0, 0])
        self.test_qc._QCSim__swap(1, 2)
        state_superposition = (1/np.sqrt(2)) * np.array([0, 0, 0, 0, -1, 1, 0, 0])
        np.testing.assert_array_almost_equal(self.test_qc.quantum_reg(), state_superposition)

class QCSimFailure(unittest.TestCase):

    def setUp(self):
        # Test machine
        self.test_qc = sq.QCSim()

        # Test program
        self.test_program = sq.Program()

    def test_instr_empty_register(self):
        '''
        The private self.__instr() method must fail when no quantum_reg indicies
        are specified to operate on.
        '''

        i_gate = gt.I(0)[0]
        self.assertRaises(TypeError, self.test_qc._QCSim__instr, i_gate)

    def test_instr_negative_loc(self):
        '''
        The private self.__instr() method must fail when specified register
        location is negative.
        '''

        i_gate = gt.I(0)[0]
        self.assertRaises(ValueError, self.test_qc._QCSim__instr, i_gate, -1)

    def test_instr_non_int_loc(self):
        '''
        The private self.__instr() method must fail when register location
        isn't integer.
        '''

        i_gate = gt.I(0)[0]
        self.assertRaises(TypeError, self.test_qc._QCSim__instr, i_gate, 1.1)

    def test_gate_and_reg_mismatch(self):
        '''
        The private self.__instr() method must fail when the number of qubit
        registers dont match the size of gate.
        '''

        i_gate = gt.I(0)[0]
        self.assertRaises(sqerr.WrongShapeError, self.test_qc._QCSim__instr, i_gate, 0, 1)

    def test_duplicate_q_reg_locs(self):
        '''
        The private self.__instr() method must fail when duplicate
        operational register locations are specified.
        '''

        i_gate = gt.I(0)[0]
        x_gate = gt.X(0)[0]
        i_x_gate_product = i_gate.gate_product(x_gate)

        self.assertRaises(ValueError, self.test_qc._QCSim__instr, i_x_gate_product, 1, 1)

    def test_swap_non_int_input(self):
        '''
        The private self.__swap() method should fail with non-integer input.
        '''

        self.assertRaises(TypeError, self.test_qc._QCSim__swap, *['peas', []])

    def test_negative_reg_loc(self):
        '''
        The private self.__measure() method must fail with negative registers.
        '''

        self.assertRaises(ValueError, self.test_qc._QCSim__measure, -1)
        self.assertRaises(ValueError, self.test_qc._QCSim__measure, 0, -1)

    def test_swap_index_out_of_range(self):
        '''
        The private self.__swap() method must fail if one of the target qubits
        is uninitialized.
        '''

        # Prepare the state |001>
        i_gate = gt.I(0)[0]
        x_gate = gt.X(0)[0]

        self.test_qc._QCSim__instr(i_gate, 2)
        self.test_qc._QCSim__instr(i_gate, 0)

        self.assertRaises(ValueError, self.test_qc._QCSim__swap, 0, 3)



if __name__ == '__main__':
    unittest.main()
