# Standard modules
import unittest
import numpy as np
import copy

# pypSQUEAK modules
# import pypsqueak.gates as gt
from pypsqueak.gates import I, X, Z, H, CNOT
from pypsqueak.api import qReg, qOp
from pypsqueak.errors import (IllegalCopyAttempt, IllegalRegisterReference,
                              WrongShapeError)


class qRegSuccess(unittest.TestCase):

    def setUp(self):
        # Test register and operator
        self.test_reg = qReg()
        self.test_op = qOp()

    def test_known_measurement_results(self):
        '''
        Verifies that the proper post-measurement state occurs in several
        cases.
        '''

        # Measure |0> correctly
        self.assertEqual(0, self.test_reg.measure(0))
        np.testing.assert_array_equal(np.array([1, 0]),
                                      self.test_reg.dump_state())

        # Measure |01> correctly
        self.test_reg += 1
        X.on(self.test_reg, 0)
        self.assertEqual(1, self.test_reg.measure(0))
        np.testing.assert_array_equal(np.array([0, 1, 0, 0]),
                                      self.test_reg.dump_state())

        # Now let's measure the observable X on the
        # superposition state (|00> - |01>)/sqrt(2).
        H.on(self.test_reg, 0)
        self.assertEqual(-1, self.test_reg.measure_observable(I.kron(X)))

    def test_measurement_collapses_register_state(self):
        '''
        Check that a ``qReg`` in the normalized version of the state
        |00> + |10> correctly collapses on measurement of qubit 0 to either
        |00> or |10>.
        '''
        initiallySuperposedRegister = qReg(2)
        H.on(initiallySuperposedRegister, 1)
        measurement_outcome = initiallySuperposedRegister.measure(1)

        if measurement_outcome == 0:
            np.testing.assert_array_equal(
                initiallySuperposedRegister.dump_state(), [1, 0, 0, 0])
        else:
            np.testing.assert_array_equal(
                initiallySuperposedRegister.dump_state(), [0, 0, 1, 0])

    def test_no_target_size_match(self):
        '''
        No targets should be necessary for a ``qOp`` acting on a ``qReg`` of
        the same size.
        '''

        self.test_op.on(self.test_reg)
        np.testing.assert_array_equal(self.test_reg.dump_state(),
                                      np.array([1, 0]))

    def test_measure_observable_smaller_than_reg(self):
        '''
        Verifies default behavior of ``qRef.measure_observable()`` is to prefix
        observable on the left with the identity when
        ``qReg.size() > observable.size()``.
        '''

        # Make the state 1/sqrt(2) (|100> + |101>) and then measure
        # I (x) I (x) X.
        X.on(self.test_reg, 2)
        H.on(self.test_reg, 0)
        result = self.test_reg.measure_observable(X)

        state_hadamard = np.zeros(8)
        state_hadamard[4] = 1/np.sqrt(2)
        state_hadamard[5] = 1/np.sqrt(2)

        np.testing.assert_array_almost_equal(state_hadamard,
                                             self.test_reg.dump_state())
        self.assertEqual(1, result)

    def test_operator_overloading(self):
        '''
        Tests that operator overloading behaves correctly for ``qReg`` objects.
        '''

        temp_reg = qReg()
        X.on(temp_reg)
        temp_reg += 1
        self.test_reg *= temp_reg
        state_001 = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(state_001, self.test_reg.dump_state())

        temp_reg = qReg()
        X.on(temp_reg)
        a_new_reg = self.test_reg * temp_reg
        state_0011 = np.zeros(16)
        state_0011[3] = 1
        np.testing.assert_array_equal(state_0011, a_new_reg.dump_state())


class qRegFailure(unittest.TestCase):

    def setUp(self):
        # Test register and operator
        self.test_reg = qReg()
        self.test_op = qOp()

    def test_copy_attempt(self):
        '''
        Verifies that copy attempts on a ``qReg`` object fail.
        '''

        self.assertRaises(IllegalCopyAttempt, copy.copy, self.test_reg)
        self.assertRaises(IllegalCopyAttempt, copy.deepcopy, self.test_reg)

    def test_qReg_construction_fails_with_non_integer_creation_arg(self):
        '''
        Verifies that ``qReg`` initialization fails with non-integer
        ``n_qubits``.
        '''

        self.assertRaises(TypeError, qReg, 1.1)

    def test_qReg_construction_fails_with_creation_arg_less_than_one(self):
        '''
        Verifies that ``qReg`` initialization fails with ``n_qubits`` less than
        one.
        '''

        self.assertRaises(ValueError, qReg, 0)
        self.assertRaises(ValueError, qReg, -1)

    def test_mult_checks_both_regs_for_dereference(self):
        '''
        Verifies that multiplication checks whether both argument registers are
        dereferenced. Added in pypSQUEAK 2.0.1.
        '''

        # Produce an active and dereferenced register (b and a, respectively).
        a = qReg()
        b = a * qReg()

        self.assertRaises(IllegalRegisterReference, a.__mul__, b)
        self.assertRaises(IllegalRegisterReference, b.__mul__, a)
        self.assertRaises(IllegalRegisterReference, a.__imul__, b)
        self.assertRaises(IllegalRegisterReference, b.__imul__, a)

    def test_register_dereferencing(self):
        '''
        Verifies that ``qReg`` instances get dereferenced in cases where
        the no-cloning theorem would be violated.
        '''

        # Multiplication operation dereferences register.
        a = qReg()
        X.on(a)
        b = qReg()
        c = a * b
        d = qReg()
        d *= c
        state_010 = np.zeros(8)
        state_010[2] = 1

        # a, b, and c should all be dereferenced. D should be in |0010>
        np.testing.assert_array_equal(state_010, d.dump_state())

        deref = [a, b, c]
        for register in deref:

            # Checks that the dereferenced register is fully dead (i.e. all
            # methods called with or on it raise an exception.
            self.assertRaises(IllegalRegisterReference, register.measure, 0)
            self.assertRaises(IllegalRegisterReference,
                              register.measure_observable,
                              Z)
            self.assertRaises(IllegalRegisterReference, register.peek)
            self.assertRaises(IllegalRegisterReference, register.dump_state)
            self.assertRaises(IllegalRegisterReference, register.__iadd__, 1)
            self.assertRaises(IllegalRegisterReference,
                              register.__mul__,
                              qReg())
            self.assertRaises(IllegalRegisterReference,
                              register.__imul__,
                              qReg())
            self.assertRaises(IllegalRegisterReference, len, register)
            self.assertRaises(IllegalRegisterReference, X.on, register, 0)

    def test_bad_measurement_index(self):
        '''
        The ``qReg.measure()`` method throws an ``IndexError`` when the
        argument isn't a nonnegative integer.
        '''

        bad_locs = [2.4, 8j, -2, 'twelve']

        for loc in bad_locs:
            self.assertRaises(IndexError, self.test_reg.measure, loc)

    def test_negative_measurement_index_fails(self):
        '''
        Measurement should fail for a negative qubit location index.
        '''

        self.assertRaises(IndexError, self.test_reg.measure, -1)

    def test_measure_observable_bad_input(self):
        '''
        The ``qReg.measure_observable()`` method should raise an exception if
        called with a non ``qOp`` object.
        '''

        invalid_ops = ['a',
                       ['peas', 'and', 'more', 'peas'],
                       71,
                       np.eye(8),
                       np.eye(2)]

        for op in invalid_ops:
            self.assertRaises(TypeError, self.test_reg.measure_observable, op)


class qOpSuccess(unittest.TestCase):

    def setUp(self):
        # Test register and operator
        self.test_reg = qReg()
        self.test_op = qOp()

    def test_known_operation_results(self):
        '''
        Verifies the resulting state of several operations.
        '''

        test_results = []
        # Takes |0> to |1>
        some_reg = qReg()
        X.on(some_reg)
        test_results.append(some_reg.dump_state())

        # Takes |0> to |0001>
        some_reg = qReg()
        X.on(some_reg)
        I.on(some_reg, 3)
        test_results.append(some_reg.dump_state())

        # Takes |0> to (1/sqrt(2))(|000> - |100>)
        some_reg = qReg()
        X.on(some_reg, 2)
        H.on(some_reg, 2)
        test_results.append(some_reg.dump_state())

        expected_results = [
            np.array([0, 1]),
            np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            1/np.sqrt(2) * np.array([1, 0, 0, 0, -1, 0, 0, 0])]

        for test_pair in zip(test_results, expected_results):
            np.testing.assert_array_almost_equal(test_pair[0], test_pair[1])

    def test_add_distant_qubit(self):
        '''
        A ``qOp`` acting on a non-extant target qubit should initialize filler
        qubits in the |0> state.
        '''

        I.on(self.test_reg, 2)
        state_000 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(state_000, self.test_reg.dump_state())


class qOpFailure(unittest.TestCase):

    def setUp(self):
        # Test register and operator
        self.test_reg = qReg()
        self.test_op = qOp()

    def test_non_unitary(self):
        '''
        Checks than an exception gets thrown for non-unitary arguments in the
        initialization of a ``qOp``.
        '''

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
            self.assertRaises(TypeError, qOp, matrix)

    def test_bad_noise_input(self):
        '''
        ``qOp`` throws a ``TypeError`` if the argument of
        ``qOp.set_noise_model()`` isn't a list of matricies.
        '''

        bad_kraus_ops = [[],
                         'nope',
                         {},
                         15,
                         (),
                         np.array([2, 3]),
                         [[14, 3], 5],
                         [[[45, 2], [14, 3]], 3]]

        for ops in bad_kraus_ops:
            self.assertRaises(TypeError, qOp, np.eye(2), kraus_ops=ops)
            self.assertRaises(TypeError, self.test_op.set_noise_model, ops)

    def test_negative_index(self):
        '''
        The ``qOp.on()`` method must fail when specified ``qReg`` target
        addresses are negative.
        '''

        self.assertRaises(IndexError, self.test_op.on, self.test_reg, -1)

    def test_no_target_size_mismatch(self):
        '''
        The ``qOp.on()`` method must fail when no ``qReg`` targets
        are specified AND the operator and register aren't the same size.
        '''

        self.test_reg += 1
        self.assertRaises(IndexError, self.test_op.on, self.test_reg)

    def test_non_int_index(self):
        '''
        The ``qOp.on()`` method must fail with non-integer targets.
        '''

        self.assertRaises(IndexError, self.test_op.on, self.test_reg, 1.1)

    def test_swap_index_out_of_range(self):
        '''
        The private ``qOp.__generate_swap()`` method must fail if one of the
        targets out of range of the ``qReg``.
        '''

        self.assertRaises(IndexError,
                          self.test_op._qOp__generate_swap,
                          self.test_reg, 0, 3)

    def test_swap_non_int_input(self):
        '''
        The private ``qOp.__generate_swap()`` method must fail if any of the
        targets are not nonnegative integers.
        '''

        some_reg = qReg()
        some_reg += 3

        self.assertRaises(IndexError,
                          self.test_op._qOp__generate_swap,
                          some_reg, 0, 0.1)
        self.assertRaises(IndexError,
                          self.test_op._qOp__generate_swap,
                          some_reg, 2, 0, -1)

    def test_invalid_matrix_rep(self):
        '''
        ``qOp`` throws a ``TypeError`` if the ``matrix_rep`` used to initialize
        it isn't a tuple/list of tuples/lists, a numpy array, or if the
        elements are not numeric.
        '''

        bad_matricies = [{'mydict': 17},
                         [],
                         [(), ()],
                         4,
                         'apples',
                         [[1, 'test'], [5, (2, 4)]],
                         np.array([['train', 4], [12, 45]])]

        for matrix in bad_matricies:
            self.assertRaises(TypeError, qOp, matrix)

    def test_non_square_matrix_rep(self):
        '''
        ``qOp`` throws a ``TypeError`` if the ``matrix_rep`` is not square.
        '''

        non_square_matrix = [[0, 1], [2, 3], [3, 4]]

        self.assertRaises(TypeError, qOp, non_square_matrix)

    def test_square_matrix_not_a_power_of_2(self):
        '''
        ``qOp`` throws a ``TypeError`` if the ``matrix_rep`` is square but not
        a power of two.
        '''

        non_square_matrix = [[0, 1, 2], [2, 3, 3], [3, 4, 5]]

        self.assertRaises(TypeError, qOp, non_square_matrix)

    def test_duplicate_q_reg_locs(self):
        '''
        The ``qOp.on()`` method must fail when duplicate target qubits are
        specified.
        '''

        self.test_reg += 1
        self.assertRaises(ValueError, CNOT.on, self.test_reg, 1, 1)

    def test_gate_and_reg_mismatch(self):
        '''
        The ``qOp.on()`` method must fail when size of the ``qReg`` doesn't
        match the size of ``qOp``.
        '''

        # Too many
        self.assertRaises(WrongShapeError,
                          self.test_op.on,
                          self.test_reg, 0, 1)

        self.test_reg += 1
        # Too few
        self.assertRaises(WrongShapeError, CNOT.on, self.test_reg, 1)

    def test_known_swaps(self):
        '''
        Verifies known swaps in the private ``qOp.__generate_swap()`` method.
        '''

        # Verify that |100> gets swapped to |001>
        X.on(self.test_reg, 2)
        swap, inverse_swap = self.test_op._qOp__generate_swap(self.test_reg, 2)
        state_100 = np.zeros(8)
        state_100[1] = 1
        np.testing.assert_array_equal(state_100,
                                      np.dot(swap, self.test_reg.dump_state()))

        # Verify that |100> gets swapped to |010> with targets 1, 2
        swap, inverse_swap = self.test_op._qOp__generate_swap(self.test_reg,
                                                              1, 2)
        state_010 = np.zeros(8)
        state_010[2] = 1
        np.testing.assert_array_equal(state_010,
                                      np.dot(swap, self.test_reg.dump_state()))

        # Verify that (|010> - |011>)/sqrt(2) gets
        # swapped to (|100> - |101>)/sqrt(2) with targets 0, 2 and 0, 2, 1
        for i in range(len(self.test_reg)):
            X.on(self.test_reg, i)
        H.on(self.test_reg, 0)
        swap, inverse_swap = self.test_op._qOp__generate_swap(self.test_reg,
                                                              0, 2)
        np.testing.assert_array_equal(swap,
                                      self.test_op._qOp__generate_swap(
                                          self.test_reg, 0, 2, 1)[0])
        np.testing.assert_array_equal(swap,
                                      self.test_op._qOp__generate_swap(
                                          self.test_reg, 0, 2)[1])
        state_hadamard = np.zeros(8)
        state_hadamard[4] = 1/np.sqrt(2)
        state_hadamard[5] = -1/np.sqrt(2)
        np.testing.assert_array_almost_equal(state_hadamard,
                                             np.dot(
                                                 swap,
                                                 self.test_reg.dump_state()))


if __name__ == '__main__':
    unittest.main()
