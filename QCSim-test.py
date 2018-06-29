import QCSim
import unittest
import numpy as np
import cmath

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
        self.test_qubit = QCSim.Qubit()

    def test_normalize_valid_input(self):
        '''normalize should rescale valid new states to be unit vectors'''

        for vec in self.valid_qubits:
            # get machine epsilon for relative error
            mach_eps = np.finfo(type(vec[0])).eps

            # change state and compute inner product
            self.test_qubit.change_state(vec)
            state_dual = np.conjugate(self.test_qubit.state)

            # if inner product != 1, then test_qubit.normalize() failed
            inner_product = np.dot(self.test_qubit.state, state_dual)

            # cmath is used so we don't discard errors in the imaginary part
            norm_query = cmath.isclose(1, inner_product, rel_tol=10*mach_eps)
            self.assertEqual(True, norm_query)


class QubitInvalidInput(unittest.TestCase):

    def setUp(self):
        self.test_qubit = QCSim.Qubit()

    def test_not_list_or_tuple(self):
        '''validate_state should fail for non-list or -tuple input'''

        not_list_or_tuple = [{'key':'value'},
                             4,
                             'string']
        for candidate in not_list_or_tuple:
            self.assertRaises(TypeError, self.test_qubit.validate_state, candidate)

    def test_non_numeric_elements(self):
        '''validate_state should fail for input with non-numeric elements'''

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
            self.assertRaises(TypeError, self.test_qubit.validate_state, candidate)

    def test_wrong_length(self):
        '''validate_state should fail for input with length not even'''
        wrong_length = [[0, 1, 34],
                        [7]]
        for candidate in wrong_length:
            self.assertRaises(QCSim.WrongShapeError, self.test_qubit.validate_state, candidate)

    def test_zero_vector(self):
        '''validate_state should fail for the null vector, list, and tuple'''
        null_vectors = [[0, 0],
                        [],
                        ()]
        for null_vec in null_vectors:
            self.assertRaises(QCSim.NullVectorError, self.test_qubit.validate_state, null_vec)

class QubitProductValidInput(unittest.TestCase):

    def test_known_qubit_product(self):
        product_state_01 = np.array([0, 1, 0, 0])
        q0 = QCSim.Qubit()
        q1 = QCSim.Qubit([0, 1])
        test_product_state_01 = q0.qubit_product(q1)

        np.testing.assert_array_equal(product_state_01, test_product_state_01.state)

class QubitProductInvalidInput(unittest.TestCase):

    def test_empty_input(self):
        self.assertRaises(TypeError, QCSim.Qubit().qubit_product)

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

    def test_pauli_gates(self):
        '''Basic Pauli matricies should initialize correctly'''

        gate_x = QCSim.Gate([[0, 1],
                             [1, 0]])
        gate_y = QCSim.Gate([[0, -1j],
                             [1j, 0]])
        gate_z = QCSim.Gate([[1, 0],
                             [0, -1]])
        gate_i = QCSim.Gate()

        test_gates = [gate_x, gate_y, gate_z, gate_i]
        test_gates = zip(self.valid_gates, test_gates)

        for gate in test_gates:
            np.testing.assert_array_equal(gate[0], gate[1].state)

class GateInvalidInput(unittest.TestCase):

    def test_non_list_non_tuple(self):
        '''Gate Should fail to initialize if input not list-like of rows'''
        not_list_of_rows = [2,
                            'string',
                            ['some', 'list'],
                            None]

        for candidate in not_list_of_rows:
            self.assertRaises(ValueError, QCSim.Gate, candidate)

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
            self.assertRaises(QCSim.WrongShapeError, QCSim.Gate, candidate)

    def test_not_even(self):
        '''Gate should fail if matrix is not 2nX2n for integer n'''
        uneven_shape = [[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]

        self.assertRaises(QCSim.WrongShapeError, QCSim.Gate, uneven_shape)

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
            self.assertRaises(QCSim.NonUnitaryInputError, QCSim.Gate, matrix)

class GateProductValidInput(unittest.TestCase):

    def test_known_two_qubit_gates(self):
        '''Checks that known two-qubit gates are formed by gate_product with one arg'''
        gate_i = QCSim.Gate()
        gate_z = QCSim.Gate([[1, 0],
                             [0, -1]])
        gate_i_prod_i = gate_i.gate_product(gate_i)
        gate_i_prod_z = gate_i.gate_product(gate_z)
        gate_i_squared = QCSim.Gate([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
        gate_i_times_z = QCSim.Gate([[1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, -1]])
        prod_value_pairs = [(gate_i_prod_i.state, gate_i_squared.state),
                            (gate_i_prod_z.state, gate_i_times_z.state)]

        for test, result in prod_value_pairs:
            np.testing.assert_array_equal(test, result)

    def test_two_args_identity(self):
        '''Checks that the tensor product of two identities with identity works'''

        gate_i = QCSim.Gate()
        gate_i_cubed = gate_i.gate_product(gate_i, gate_i)
        gate_should_equal = QCSim.Gate(np.eye(8).tolist())

        np.testing.assert_array_equal(gate_i_cubed.state, gate_should_equal.state)

class GateProductInvalidInput(unittest.TestCase):

    def test_empty_product(self):
        '''gate_product should fail with no arguments'''
        self.assertRaises(TypeError, QCSim.Gate().gate_product)

    def test_non_gate_input(self):
        '''gate_product should fail with non Gate() input.'''
        not_gates = [2, 16, [1, 2], [[1, 2], [2, 3]], []]

        for candidate in not_gates:
            self.assertRaises(TypeError, QCSim.Gate().gate_product, candidate)


if __name__ == '__main__':
    unittest.main()
