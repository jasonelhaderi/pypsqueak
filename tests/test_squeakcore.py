import cmath
import pytest
import numpy as np

from pypsqueak.squeakcore import Qubit, Gate
from pypsqueak.errors import WrongShapeError, NullVectorError


class TestQubitValidInput:
    @pytest.mark.parametrize('initialization_vector', [
        [1, 0, 1, 1],
        [2, 14 - 8j],
        (0, 1, 0, 0, 0, 0, 0, 1)
    ])
    def test_change_and_initialize_equiv(self, initialization_vector, benchmark):
        '''
        Initalization of a ``Qubit`` as well as the ``Qubit.change_state()`` method
        should result in the same state if handed the same argument.
        '''
        test_qubit = Qubit()
        benchmark(test_qubit.change_state, initialization_vector)
        assert np.array_equal(
            test_qubit.state(),
            Qubit(initialization_vector).state())

    @pytest.mark.parametrize('valid_qubit', [
        [1.0, 0],
        [0.0, 1],
        [1.0j, 0],
        [1.0j/np.sqrt(2), -1j/np.sqrt(2)],
        [24.0, 6],
        (24.0, 6),
        [1 - 1j, 0j],
        [-25.0, 9j],
        [-25.0, 9j, 3, 5],
        np.array([1.0, 0])])
    def test_normalize_valid_input(self, valid_qubit):
        '''
        ``Qubit.change_state()`` should rescale valid new states to unit vectors.
        '''
        test_qubit = Qubit()
        mach_eps = np.finfo(type(valid_qubit[0])).eps

        state_dual = np.conjugate(test_qubit.state())
        inner_product = np.dot(test_qubit.state(), state_dual)

        assert cmath.isclose(1, inner_product, rel_tol=10**mach_eps)

    def test_qubit_product_one_arg(self, benchmark):
        '''
        Verifies proper result for one arg in ``Qubit.qubit_product()``.
        '''
        test_qubit = Qubit()
        expected_product = np.array([0, 1, 0, 0])
        q1 = Qubit([0, 1])
        actual_product = benchmark(test_qubit.qubit_product, q1)

        assert np.array_equal(actual_product.state(), expected_product)

    def test_qubit_product_two_args(self, benchmark):
        '''
        Verifies proper results for two args in ``Qubit.qubit_product()``.
        '''

        expected_product = np.zeros(8, dtype=np.cdouble)
        expected_product[0] = -1j

        initial_state = Qubit([1j, 0])
        actual_product = benchmark(
            initial_state.qubit_product,
            Qubit([1j, 0]),
            Qubit([1j, 0]))

        np.array_equal(actual_product.state(), expected_product)

    def test_computational_decomp_two_qubits(self, benchmark):
        '''
        Checks that ``Qubit.computational_decomp()`` is correct for a Bell pair.
        '''

        bell_pair = Qubit([1, 0, 0, 1])
        expected_decomposition = {
            '00': 1/np.sqrt(2),
            '01': 0,
            '10': 0,
            '11': 1/np.sqrt(2)
        }

        actual_decomp = benchmark(bell_pair.computational_decomp)

        assert expected_decomposition == actual_decomp

    def test_computational_decomp_three_qubits(self, benchmark):
        '''
        Checks that ``Qubit.computational_decomp()`` is correct for a three
        qubit state.
        '''

        some_qubits = Qubit([1, 0, 1, 0, 0, 0, 0, 1])
        expected_decomposition = {
            '000': 1/np.sqrt(3),
            '001': 0,
            '010': 1/np.sqrt(3),
            '011': 0,
            '100': 0,
            '101': 0,
            '110': 0,
            '111': 1/np.sqrt(3)
        }

        actual_decomp = benchmark(some_qubits.computational_decomp)

        assert expected_decomposition == actual_decomp

    def test_string_rep_bell_state(self):
        '''
        Checks that ``str(Qubit)`` is correct for a Bell state.
        '''

        bell_pair = Qubit([1, 0, 0, -1])
        expected_rep = '(7.07e-01)|00> + (-7.07e-01)|11>'

        assert expected_rep == str(bell_pair)

    def test_string_rep_first_term_negative(self):
        '''
        Checks that ``str(Qubit)`` is correct for a state with a negative first term.
        '''

        qubit = Qubit([-1, 0, 0, 0, 0, 0, 2, 1])
        expected_rep = '(-4.08e-01)|000> + (8.16e-01)|110> + (4.08e-01)|111>'

        assert expected_rep == str(qubit)

    def test_string_rep_later_term_negative(self):
        '''
        Checks that ``str(Qubit)`` is correct for a state with a term other than the first negative.
        '''
        qubit = Qubit([1, 0, 0, 0, 0, 0, -2, 1])
        expected_rep = '(4.08e-01)|000> + (-8.16e-01)|110> + (4.08e-01)|111>'

        assert expected_rep == str(qubit)

    def test_string_rep_complex_term(self):
        '''
        Checks that ``str(Qubit)`` is correct for a state with a complex term.
        '''

        qubit = Qubit([1 - 1j, 1j])
        expected_rep = '(5.77e-01-5.77e-01j)|0> + (5.77e-01j)|1>'

        assert expected_rep == str(qubit)


class TestQubitInvalidInput:

    @pytest.mark.parametrize('bad_vector', [
        {'key': 'value'},
        4,
        'string'
    ])
    def test_not_list_or_tuple(self, bad_vector):
        '''
        ``Qubit.change_state()`` should fail when called on a non-listlike.
        '''
        test_qubit = Qubit()
        with pytest.raises(TypeError):
            test_qubit.change_state(bad_vector)

    @pytest.mark.parametrize('bad_vector', [
        ['string'],
        [['string'], ['string']],
        [[34, 25]],
        [[24], [25]],
        ([24], (25)),
        [[24, 35], [-2]],
        [[-1, 3], [4, 17], 2],
        [{'key': 'value'}, {'key': 'value'}],
        [(24, 35), (-2)]])
    def test_non_numeric_elements(self, bad_vector):
        '''
        ``Qubit.change_state()`` should fail for a list-like with nonnumeric
        elements.
        '''
        test_qubit = Qubit()
        with pytest.raises(TypeError):
            test_qubit.change_state(bad_vector)

    @pytest.mark.parametrize('bad_vector', [
        [0, 1, 34],
        [7]])
    def test_wrong_length(self, bad_vector):
        '''
        ``Qubit.change_state()`` should fail for a vector with length that
        isn't a power of 2 > 1.
        '''
        test_qubit = Qubit()

        with pytest.raises(WrongShapeError):
            test_qubit.change_state(bad_vector)

    @pytest.mark.parametrize('null_vector', [
        [0, 0],
        [],
        ()])
    def test_zero_vector(self, null_vector):
        '''
        ``Qubit.change_state()`` should fail for null vectors.
        '''
        test_qubit = Qubit()
        with pytest.raises(NullVectorError):
            test_qubit.change_state(null_vector)

    def test_empty_input_qubit_product(self):
        '''
        ``Qubit.qubit_product()`` with empty argument should raise a
        ``TypeError``.
        '''
        with pytest.raises(TypeError):
            Qubit().qubit_product()


class TestGateValidInput:
    @pytest.mark.parametrize('pauli_matrix', [
        np.array([[0, 1],
                  [1, 0]]),
        np.array([[0, -1j],
                  [1j, 0]]),
        np.array([[1, 0],
                  [0, -1]]),
        np.array([[1, 0],
                  [0, 1]])
    ])
    def test_pauli_gates_initialize(self, pauli_matrix, benchmark):
        '''
        Basic Pauli matricies should initialize correctly.
        '''
        result_gate = benchmark(Gate, pauli_matrix)
        assert np.array_equal(pauli_matrix, result_gate.state())

class TestGateInvalidInput:
    @pytest.mark.parametrize('not_list_of_rows', [
        2,
        'string',
        ['some', 'list'],
        None])
    def test_non_list_non_tuple(self, not_list_of_rows):
        '''
        ``Gate`` should fail to initialize if input not list-like of rows.
        '''
        with pytest.raises(TypeError):
            Gate(not_list_of_rows)

    @pytest.mark.parametrize('nonnumeric_matrix', [
        [[np.array([3, 4]), 6],
         [1, -1j]],
        (('item', 3),
         (3, 5))
    ])
    def test_non_numeric_elements(self, nonnumeric_matrix):
        '''
        ``Gate`` should fail to initialize if input elements aren't numeric.
        '''
        with pytest.raises(TypeError):
            Gate(nonnumeric_matrix)

    def test_empty_list_input(self):
        '''
        ``Gate`` should fail to initialize with the empty list as an input.
        '''
        with pytest.raises(TypeError):
            Gate([])

    @pytest.mark.parametrize('non_square_matrix', [
        [[]],
        [[1, 2]],
        [[1, 2, 3],
         [1, 3, 4]],
        [[0, 0],
         [0, 0, 1]],
        ([1, 2],
         (1, 2, 3))])
    def test_not_square(self, non_square_matrix):
        '''
        ``Gate`` should throw ``TypeError`` if input matrix doesn't have shape
        ``(n, n)``.
        '''
        with pytest.raises(TypeError):
            Gate(non_square_matrix)

    def test_not_power_2(self):
        '''
        ``Gate`` should fail if matrix is not ``n``X``n`` for ``n == 1`` or
        ``n`` a power of 2.
        '''
        uneven_shape = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
        with pytest.raises(TypeError):
            Gate(uneven_shape)

    @pytest.mark.parametrize('non_unitary_matrix', [
        [[1, 5],
         [10, 7]],
        [[0, 1j],
         [1j + 17, 0]],
        [[4, 0],
         [0, -3]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]])
    def test_not_unitary(self, non_unitary_matrix):
        '''
        ``Gate`` should fail to initialize non-unitary matrix.
        '''
        with pytest.raises(TypeError):
            Gate(non_unitary_matrix)


class TestGateProductValidInput:
    @pytest.mark.parametrize('gate_1, gate_2, expected_product_gate', [
        (
            Gate(),
            Gate([[1, 0],
                  [0, -1]]),
            Gate([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, -1]])),  # I times Z
        (
            Gate(),
            Gate(),
            Gate([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]))  # I times I
    ])
    def test_known_two_qubit_gates(self, gate_1, gate_2, expected_product_gate, benchmark):
        '''
        Checks that known two-qubit gates are formed by ``Gate.gate_product()``
        with one arg.
        '''
        actual_product_gate = benchmark(gate_1.gate_product, gate_2)
        assert np.array_equal(actual_product_gate.state(), expected_product_gate.state())

    def test_empty_product(self):
        '''
        ``Gate.gate_product()`` should return ``self`` with no arguments.
        '''
        gate_z = Gate([[1, 0],
                       [0, -1]])
        gate_empty_arg = gate_z.gate_product()

        assert np.array_equal(gate_z.state(), gate_empty_arg.state())

    def test_two_args_identity(self):
        '''
        Checks that the tensor product of two identities with identity works.
        '''
        gate_i = Gate()
        gate_i_cubed = gate_i.gate_product(gate_i, gate_i)
        gate_should_equal = Gate(np.eye(8))

        assert np.array_equal(gate_i_cubed.state(), gate_should_equal.state())

    def test_gate_product_5_single_qubit_gates(self, benchmark):
        '''
        Check that a nontrivial 5 gate product is computed correctly.
        '''
        pauli_x = Gate([[0, 1],
                        [1, 0]])
        pauli_y = Gate([[0, -1j],
                        [1j, 0]])
        result_gate = benchmark(
            pauli_x.gate_product,
            pauli_y,
            pauli_y,
            pauli_y,
            pauli_y)

        # Product is only nonzero on reverse diagonal
        expected_gate = np.fliplr(
            np.diag([1, -1, -1, 1,
                     -1, 1, 1, -1,
                     -1, 1, 1, -1,
                     1, -1, -1, 1,
                     1, -1, -1, 1,
                     -1, 1, 1, -1,
                     -1, 1, 1, -1,
                     1, -1, -1, 1])
        )
        assert np.array_equal(expected_gate, result_gate.state())

    def test_gate_product_between_1_and_2_qubit_gates(self, benchmark):
        '''
        Check that a nontrivial 3 qubit gate product gets computed correctly.
        '''
        pauli_x = Gate([[0, 1],
                        [1, 0]])
        pauli_yz = Gate([[0, 0, -1j, 0],
                         [0, 0, 0, 1j],
                         [1j, 0, 0, 0],
                         [0, -1j,0, 0]])
        expected_gate = np.array([[0, 0, 0, 0, 0, 0, -1j, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1j],
                                  [0 ,0, 0, 0, 1j, 0, 0, 0],
                                  [0, 0, 0, 0, 0, -1j, 0, 0],
                                  [0, 0, -1j, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1j, 0, 0, 0, 0],
                                  [1j, 0, 0, 0, 0, 0, 0, 0],
                                  [0, -1j, 0, 0, 0, 0, 0, 0]])
        result_gate = benchmark(pauli_x.gate_product, pauli_yz)
        assert np.array_equal(expected_gate, result_gate.state())


class TestGateProductInvalidInput:

    @pytest.mark.parametrize('bad_gate', [
        2,
        16,
        [1, 2],
        [[1, 2],
         [2, 3]],
        []
    ])
    def test_non_gate_input(self, bad_gate):
        '''
        ``Gate.gate_product()`` should fail with non ``Gate`` input.
        '''
        with pytest.raises(TypeError):
            Gate().gate_product(bad_gate)
