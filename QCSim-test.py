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
                    [-25.0, 9j]]

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
        '''validate_state should fail for input with length != 2'''
        wrong_length = [[0, 1, 34],
                        [7],
                        [4, 3, 10, -43]]
        for candidate in wrong_length:
            self.assertRaises(QCSim.WrongShapeError, self.test_qubit.validate_state, candidate)

    def test_zero_vector(self):
        '''validate_state should fail for the null vector, list, and tuple'''
        null_vectors = [[0, 0],
                        [],
                        ()]
        for null_vec in null_vectors:
            self.assertRaises(QCSim.NullVectorError, self.test_qubit.validate_state, null_vec)

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

    def test_wrong_shape(self):
        '''Gate should fail to initialize if input doesn't have shape (2, 2)'''
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


# TensorProduct tests

class TensorProductInvalidInput(unittest.TestCase):

    def test_empty_input(self):
        '''Initialization should fail with empty input'''
        self.assertRaises(TypeError, QCSim.TensorProduct)

    def test_input_not_qubit_or_gate(self):
        '''Input with non-qubit or -gate elements should fail to initialize'''

        wrong_elements = [[1],
                          [[2, 5]],
                          ['qwe'],
                          ((4)),
                          {'key':'value'},
                          2,
                          'string',
                          *(2, 3, 5)]
        for candidate in wrong_elements:
            self.assertRaises(TypeError, QCSim.TensorProduct, candidate)

    def test_inhomogenous_input(self):
        '''TensorProduct should fail to initialize with inhomogenous input'''

        q = QCSim.Qubit()
        g = QCSim.Gate()

        self.assertRaises(QCSim.InhomogenousInputError, QCSim.TensorProduct, *(q, g))

class TensorProductValidInput(unittest.TestCase):

    def setUp(self):
        self.test_q0 = QCSim.Qubit()
        self.test_q1 = QCSim.Qubit([0, 1])

    def test_gate_product(self):
        '''Verifies the tensor product of X and I'''
        gate_x = QCSim.Gate([[0, 1],
                             [1, 0]])
        gate_i = QCSim.Gate()
        test_product = QCSim.TensorProduct(gate_x, gate_i)
        actual_product = [np.array([[0,1], [1,0]]), np.array([[1,0], [0,1]])]
        product_comparison = zip(actual_product, test_product.parts)
        for gates in product_comparison:
            np.testing.assert_array_equal(gates[0], gates[1].state)

    def test_qubit_product(self):
        '''Verifies the tensor product of |0> and |1>'''

        product_state_01 = [self.test_q0.state, self.test_q1.state]
        test_product_state_01 = [qubit.state for qubit in QCSim.TensorProduct(self.test_q0, self.test_q1).parts]
        self.assertEqual(product_state_01, test_product_state_01)


if __name__ == '__main__':
    unittest.main()
