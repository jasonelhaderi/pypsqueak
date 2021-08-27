import copy
import numpy as np
import pytest
from pypsqueak.gates import I, X, Z, H, CNOT
from pypsqueak.api import qReg, qOp, qOracle
from pypsqueak.errors import (IllegalCopyAttempt, IllegalRegisterReference,
                              WrongShapeError, NonUnitaryInputError)
from pypsqueak.noise import damping_map


class TestqRegSuccess:
    def test_create_10qb_qreg(self, benchmark):
        '''
        Checks that a `qReg` instance with 10 qubits is created without error.
        Benchmarks time.
        '''
        q = benchmark(qReg, 10)

        assert len(q) == 10

    def test_measure_qubit_0(self, benchmark):
        '''
        Verifies that |0> is measured correctly.
        '''
        test_reg = qReg()
        result = benchmark(test_reg.measure, 0)
        assert result == 0
        assert np.array_equal(np.array([1, 0]), test_reg.dump_state())

    def test_measure_qubits_01(self):
        '''
        Verifies that |01> is measured correctly.
        '''
        test_reg = qReg(2)
        X.on(test_reg, 0)
        assert test_reg.measure(0) == 1
        assert test_reg.measure(1) == 0
        assert np.array_equal(np.array([0, 1, 0, 0]), test_reg.dump_state())

    def test_measure_X_in_bell_state(self, benchmark):
        '''
        Verifies that the observable X has the correct value in the Bell state
        (|00> - |01>)/sqrt(2).
        '''
        test_reg = qReg(2)
        X.on(test_reg, 0)
        H.on(test_reg, 0)
        result = benchmark(test_reg.measure_observable, I.kron(X))
        assert np.isclose(result, -1)

    def test_measurement_collapses_two_qubit_register(self, benchmark):
        '''
        Check that a ``qReg`` in the normalized version of the state
        |00> + |10> correctly collapses on measurement of qubit 0 to either
        |00> or |10>.
        '''
        superposition_state = qReg(2)
        H.on(superposition_state, 1)
        measurement_outcome = benchmark(superposition_state.measure, 1)
        collapsed_to_00 = (np.array_equal(np.array([1, 0, 0, 0]), superposition_state.dump_state())
                            and measurement_outcome == 0)
        collapsed_to_10 = (np.array_equal(np.array([0, 0, 1, 0]), superposition_state.dump_state())
                            and measurement_outcome == 1)

        assert collapsed_to_00 or collapsed_to_10

    def test_measurement_on_five_qubit_state(self, benchmark):
        '''
        Checks measurement of qubits 3 and 4 in (|01001> + |01011>)/sqrt(2).
        '''
        test_reg = qReg(5)
        X.on(test_reg, 3)
        X.on(test_reg, 0)
        H.on(test_reg, 1)

        assert benchmark(test_reg.measure, 3) == 1
        assert test_reg.measure(4) == 0

    def test_no_target_size_match(self):
        '''
        No targets should be necessary for a ``qOp`` acting on a ``qReg`` of
        the same size.
        '''
        test_reg = qReg()
        test_op = qOp()
        test_op.on(test_reg)
        assert np.array_equal(test_reg.dump_state(), np.array([1, 0]))

    def test_measure_2qb_observable_on_3qb_reg(self, benchmark):
        '''
        Verifies default behavior of ``qRef.measure_observable()`` is to prefix
        observable on the left with the identity when
        ``qReg.size() > observable.size()``.
        '''
        test_reg = qReg()

        # Make the state 1/sqrt(2) (|100> + |101>) and then measure
        # I (x) I (x) X.
        X.on(test_reg, 2)
        H.on(test_reg, 0)
        result = benchmark(test_reg.measure_observable, X)

        state_hadamard = np.zeros(8)
        state_hadamard[4] = 1/np.sqrt(2)
        state_hadamard[5] = 1/np.sqrt(2)

        assert np.allclose(state_hadamard, test_reg.dump_state())
        assert np.isclose(result, 1)

    def test_operator_overloading_misc(self):
        '''
        Tests that several operator overloading methods behave correctly
        for ``qReg`` objects.
        '''
        test_reg = qReg()
        temp_reg = qReg()
        X.on(temp_reg)
        temp_reg += 1
        test_reg *= temp_reg
        state_001 = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        assert np.array_equal(state_001, test_reg.dump_state())

        temp_reg = qReg()
        X.on(temp_reg)
        a_new_reg = test_reg * temp_reg
        state_0011 = np.zeros(16)
        state_0011[3] = 1
        assert np.array_equal(state_0011, a_new_reg.dump_state())

    def test_operator_overloading_iadd(self):
        '''
        Tests that `+=` adds one or more qubits to register.
        '''

        q = qReg()
        q += 3

        assert len(q) == 4

    def test_operator_overloading_imul_dereferences_arg(self):
        '''
        Checks that `*=` dereferences the right hand operand.
        '''

        q = qReg()
        p = qReg()

        q *= p

        assert p._qReg__is_dereferenced and not q._qReg__is_dereferenced


class TestqRegFailure:
    @pytest.fixture
    def dereferenced_reg(self):
        '''
        Provides a dereferenced register.
        '''
        q = qReg()
        q * qReg()

        return q

    @pytest.mark.parametrize("qreg_copy_attempt", [
        copy.copy,
        copy.deepcopy
    ])
    def test_copy_attempt_fails(self, qreg_copy_attempt):
        '''
        Verifies that copy attempts on a ``qReg`` object fail.
        '''
        with pytest.raises(IllegalCopyAttempt):
            test_reg = qReg()
            qreg_copy_attempt(test_reg)

    @pytest.mark.parametrize("qreg_initializer", [
        1.1,
        '0',
    ])
    def test_qReg_fails_with_nonint_creation_arg(self, qreg_initializer):
        '''
        Verifies that ``qReg`` initialization fails with non-integer
        ``n_qubits``.
        '''
        with pytest.raises(TypeError):
            qReg(qreg_initializer)

    @pytest.mark.parametrize("qreg_initializer", [
        0,
        -1,
    ])
    def test_qReg_fails_with_nonpositive_int_creation_arg(self, qreg_initializer):
        '''
        Verifies that ``qReg`` initialization fails with non-integer
        ``n_qubits``.
        '''
        with pytest.raises(ValueError):
            qReg(qreg_initializer)

    def test_mult_checks_both_regs_for_dereference(self, dereferenced_reg):
        '''
        Verifies that multiplication checks whether both argument registers are
        dereferenced.
        '''

        a = qReg()
        b = dereferenced_reg
        with pytest.raises(IllegalRegisterReference):
            a * b

        with pytest.raises(IllegalRegisterReference):
            b * a

        with pytest.raises(IllegalRegisterReference):
            a *= b

        with pytest.raises(IllegalRegisterReference):
            b *= a

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
        assert np.array_equal(state_010, d.dump_state())

        deref = [a, b, c]
        for register in deref:

            # Checks that the dereferenced register is fully dead (i.e. all
            # methods called with or on it raise an exception.
            with pytest.raises(IllegalRegisterReference):
                register.measure(0)
            with pytest.raises(IllegalRegisterReference):
                register.measure_observable(Z)
            with pytest.raises(IllegalRegisterReference):
                register.peek()
            with pytest.raises(IllegalRegisterReference):
                register.dump_state()
            with pytest.raises(IllegalRegisterReference):
                register += 1
            with pytest.raises(IllegalRegisterReference):
                register *= qReg()
            with pytest.raises(IllegalRegisterReference):
                register * qReg()
            with pytest.raises(IllegalRegisterReference):
                len(register)
            with pytest.raises(IllegalRegisterReference):
                X.on(register, 0)

    @pytest.mark.parametrize('bad_index', [
        2.4,
        8j,
        -2,
        'twelve'
    ])
    def test_bad_measurement_index(self, bad_index):
        '''
        The ``qReg.measure()`` method throws an ``IndexError`` when the
        argument isn't a nonnegative integer.
        '''
        test_reg = qReg()

        with pytest.raises(IndexError):
            test_reg.measure(bad_index)

    @pytest.mark.parametrize('bad_observable', [
        'a',
        ['peas', 'and', 'more', 'peas'],
        71,
        np.eye(8),
        np.eye(2)
    ])
    def test_measure_observable_bad_input(self, bad_observable):
        '''
        The ``qReg.measure_observable()`` method should raise an exception if
        called with a non ``qOp`` object.
        '''
        test_reg = qReg()

        with pytest.raises(TypeError):
            test_reg.measure_observable(bad_observable)

    def test__generateStateTransitionProbabilities(self):
        '''
        Checks that a ``NonUnitaryInputError`` is thrown for nonunitary
        arguments.
        '''
        nonUnitaryMatrix = np.eye(4)
        nonUnitaryMatrix[0, 0] = 0

        twoQubitRegister = qReg(2)

        with pytest.raises(NonUnitaryInputError):
            twoQubitRegister._generateStateTransitionProbabilities(
                nonUnitaryMatrix)

    @pytest.mark.parametrize('bad_operand', [
        3.2,
        -2
    ])
    def test_overload_iadd_fails_for_neg_or_non_int(self, bad_operand):
        '''
        Checks that a ValueError is thrown for noninteger right hand operand
        to `+=`.
        '''
        q = qReg()
        with pytest.raises(ValueError):
            q += bad_operand

    def test_operator_overloading_imul_fails_for_non_qreg_arg(self):
        '''
        Checks that `*=` throws a TypeError when the right hand operand is
        not a `qReg`.
        '''

        q = qReg()
        with pytest.raises(TypeError):
            q *= 4

    def test_operator_overloading_imul_on_dereferenced_args_fails(self, dereferenced_reg):
        '''
        Checks that `*=` fails when either involved register is dereferenced.
        '''
        test_reg = qReg()
        dead_reg = dereferenced_reg

        with pytest.raises(IllegalRegisterReference):
            test_reg *= dead_reg

        with pytest.raises(IllegalRegisterReference):
            dead_reg *= test_reg


class TestqOpSuccess:
    @staticmethod
    def apply_gate_to_zero_reg_dump_state(
            num_qubits: int,
            gate: qOp,
            target: int,
            initializer: qOp = None) -> np.array:
        q = qReg()
        if initializer is not None:
            initializer.on(q)
        gate.on(q, target)

        return q.dump_state()

    @pytest.mark.parametrize("q_circuit, expected_result", [
        (
            lambda q_reg: X.on(q_reg),
            np.array([0, 1])
        ),  #  Takes |0> to |1>
        (
            lambda q_reg: X.on(q_reg) or I.on(q_reg, 3),
            np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ),  # Takes |0> to |0001>
        (
            lambda q_reg: X.on(q_reg, 2) or H.on(q_reg, 2),
            1/np.sqrt(2) * np.array([1, 0, 0, 0, -1, 0, 0, 0])
        )  # Takes |0> to (1/sqrt(2))(|000> - |100>)
    ])
    def test_known_operation_results(self, q_circuit, expected_result):
        '''
        Verifies the resulting state of several sets of operations.
        '''
        some_reg = qReg()
        q_circuit(some_reg)
        assert np.allclose(some_reg.dump_state(), expected_result)

    def test_add_distant_qubit(self, benchmark):
        '''
        A ``qOp`` acting on a non-extant target qubit should initialize filler
        qubits in the |0> state.
        '''
        test_reg = qReg()
        benchmark(I.on, test_reg, 2)
        # I.on(test_reg, 2)
        state_000 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        assert np.array_equal(state_000, test_reg.dump_state())

    def test_known_swaps(self):
        '''
        Verifies known swaps in the private ``qOp.__generate_swap()`` method.
        '''
        test_reg = qReg()
        test_op = qOp()
        # Verify that |100> gets swapped to |001>
        X.on(test_reg, 2)
        swap, inverse_swap = test_op._qOp__generate_swap(test_reg, 2)
        state_100 = np.zeros(8)
        state_100[1] = 1
        assert np.array_equal(state_100,
                              np.dot(swap, test_reg.dump_state()))

        # Verify that |100> gets swapped to |010> with targets 1, 2
        swap, inverse_swap = test_op._qOp__generate_swap(test_reg,
                                                         1, 2)
        state_010 = np.zeros(8)
        state_010[2] = 1
        assert np.array_equal(state_010,
                              np.dot(swap, test_reg.dump_state()))

        # Verify that (|010> - |011>)/sqrt(2) gets
        # swapped to (|100> - |101>)/sqrt(2) with targets 0, 2 and 0, 2, 1
        for i in range(len(test_reg)):
            X.on(test_reg, i)
        H.on(test_reg, 0)
        swap, inverse_swap = test_op._qOp__generate_swap(test_reg,
                                                         0, 2)
        assert np.array_equal(swap,
                              test_op._qOp__generate_swap(
                                  test_reg, 0, 2, 1)[0])
        assert np.array_equal(swap,
                              test_op._qOp__generate_swap(
                                  test_reg, 0, 2)[1])
        state_hadamard = np.zeros(8)
        state_hadamard[4] = 1/np.sqrt(2)
        state_hadamard[5] = -1/np.sqrt(2)
        assert np.allclose(state_hadamard,
                           np.dot(
                               swap,
                               test_reg.dump_state()))

    def test_identity_swap_for_no_targets(self):
        '''
        Verifies that the private ``qOp.__generate_swap()`` retuns identity
        matrices as permutation operators when no nargets are specified.
        '''

        two_qubits = qReg(2)
        permutation, inverse = CNOT._qOp__generate_swap(two_qubits)

        assert np.array_equal(np.eye(4, 4), permutation)
        assert np.array_equal(np.eye(4, 4), inverse)

    def test_set_noise_model(self):
        '''
        Verifies that setting a noise model succeeds.
        '''
        test_op = qOp()
        test_op.set_noise_model(damping_map(0.2))
        some_op = qOp(np.eye(2), damping_map(0.2))
        list_of_kraus_ops = [
            np.array([[1, 0],
                      [0, np.sqrt(0.8)]]),
            np.array([[0, np.sqrt(0.2)],
                      [0, 0]]),
        ]

        assert np.array_equal(
            some_op._qOp__noise_model.getKrausOperators(),
            list_of_kraus_ops
        )
        assert np.array_equal(
            test_op._qOp__noise_model.getKrausOperators(),
            list_of_kraus_ops
        )

    def test_apply_noisy_gate(self, benchmark):
        '''
        Deterministic verification of application of gate with
        some NoiseModel set (using prob 1 amplitude damping).
        '''
        test_op = qOp()
        test_op.set_noise_model(damping_map(1.0))

        registerInZeroStateInitiallyResult = benchmark(
            self.apply_gate_to_zero_reg_dump_state,
            1,
            test_op,
            0
        )
        registerInOneStateInitiallyResult = self.apply_gate_to_zero_reg_dump_state(1, test_op, 0, X)

        assert np.array_equal(
            registerInZeroStateInitiallyResult, [1, 0])
        assert np.array_equal(
            registerInOneStateInitiallyResult, [1, 0])

    def test_apply_noisy_gate_with_raised_register(self):
        '''
        Deterministic verification of application of gate with
        some NoiseModel set (using prob 1 amplitude damping) where
        qReg needs to be raised. Measuring gate.on(|01>, 1).
        '''
        test_op = qOp()
        test_op.set_noise_model(damping_map(1.0))

        singleQubitInOneStateInitiallyResult = self.apply_gate_to_zero_reg_dump_state(1, test_op, 1, X)

        assert np.array_equal(
            singleQubitInOneStateInitiallyResult, [0, 1, 0, 0])

    def test_mul_with_qOp_preserves_first_qOp_noise_model(self):
        '''
        Checks that after multiplication, the resulting `qOp` inherits
        the noise model of the first operand.
        '''
        op1 = qOp(kraus_ops=damping_map(0.3))
        op2 = qOp()

        assert np.array_equal((op1 * op2)._qOp__noise_model, damping_map(0.3))
        assert (op2 * op1)._qOp__noise_model is None
        assert (op1 * op2)._qOp__noise_model == op1._qOp__noise_model


class TestqOpFailure:
    @pytest.mark.parametrize('nonunitary_matrix', [
        [[1, 5],
         [10, 7]],
        [[0, 1j],
         [1j + 17, 0]],
        [[4, 0],
         [0, -3]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    ])
    def test_non_unitary(self, nonunitary_matrix):
        '''
        Checks than an exception gets thrown for non-unitary arguments in the
        initialization of a ``qOp``.
        '''
        with pytest.raises(TypeError):
            qOp(nonunitary_matrix)

    def test_set_noise_model_bad_input(self):
        '''
        ``qOp`` throws a ``TypeError`` if the argument of
        ``qOp.set_noise_model()`` isn't a NoiseModel object.
        '''
        test_op = qOp()
        list_of_kraus_ops = [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]
        with pytest.raises(TypeError):
            qOp(np.eye(2), kraus_ops=list_of_kraus_ops)

        with pytest.raises(TypeError):
            test_op.set_noise_model(list_of_kraus_ops)

    @pytest.mark.parametrize('bad_target', [
        -1,
        1.1,
        '0'
    ])
    def test_non_nonnegative_integer_index_fails(self, bad_target):
        '''
        The ``qOp.on()`` method must fail when specified ``qReg`` target
        addresses are negative.
        '''
        test_reg = qReg()
        test_op = qOp()
        with pytest.raises(IndexError):
            test_op.on(test_reg, bad_target)

    def test_no_target_size_mismatch(self):
        '''
        The ``qOp.on()`` method must fail when no ``qReg`` targets
        are specified AND the operator and register aren't the same size.
        '''
        test_reg = qReg()
        test_op = qOp()
        test_reg += 1
        with pytest.raises(IndexError):
            test_op.on(test_reg)

    def test_swap_index_out_of_range(self):
        '''
        The private ``qOp.__generate_swap()`` method must fail if one of the
        targets out of range of the ``qReg``.
        '''
        test_reg = qReg()
        test_op = qOp()
        with pytest.raises(IndexError):
            test_op._qOp__generate_swap(test_reg, 0, 3)

    @pytest.mark.parametrize('bad_target', [
        -1,
        1.1,
    ])
    def test_swap_non_int_input(self, bad_target):
        '''
        The private ``qOp.__generate_swap()`` method must fail if any of the
        targets are not nonnegative integers.
        '''

        some_reg = qReg()
        test_op = qOp()
        some_reg += 3
        with pytest.raises(IndexError):
            test_op._qOp__generate_swap(some_reg, 1, 0, bad_target)

    @pytest.mark.parametrize('bad_matrix', [
        {'mydict': 17},
        [],
        [(), ()],
        4,
        'apples',
        [[1, 'test'], [5, (2, 4)]],
        np.array([['train', 4], [12, 45]])
    ])
    def test_invalid_matrix_rep(self, bad_matrix):
        '''
        ``qOp`` throws a ``TypeError`` if the ``matrix_rep`` used to initialize
        it isn't a tuple/list of tuples/lists, a numpy array, or if the
        elements are not numeric.
        '''
        with pytest.raises(TypeError):
            qOp(bad_matrix)

    def test_non_square_matrix_rep(self):
        '''
        ``qOp`` throws a ``TypeError`` if the ``matrix_rep`` is not square.
        '''
        non_square_matrix = [[0, 1], [2, 3], [3, 4]]
        with pytest.raises(TypeError):
            qOp(non_square_matrix)

    def test_square_matrix_not_a_power_of_2(self):
        '''
        ``qOp`` throws a ``TypeError`` if the ``matrix_rep`` is square but not
        a power of two.
        '''
        square_but_wrong_size = [[0, 1, 2], [2, 3, 3], [3, 4, 5]]
        with pytest.raises(TypeError):
            qOp(square_but_wrong_size)

    def test_duplicate_q_reg_locs(self):
        '''
        The ``qOp.on()`` method must fail when duplicate target qubits are
        specified.
        '''
        test_reg = qReg()
        test_reg += 1
        with pytest.raises(ValueError):
            CNOT.on(test_reg, 1, 1)

    def test_gate_and_reg_mismatch(self):
        '''
        The ``qOp.on()`` method must fail when size of the ``qReg`` doesn't
        match the size of ``qOp``.
        '''
        test_reg = qReg()
        test_op = qOp()
        # Too many targets
        with pytest.raises(WrongShapeError):
            test_op.on(test_reg, 0, 1)

        test_reg += 1
        # Too few targets
        with pytest.raises(WrongShapeError):
            CNOT.on(test_reg, 1)

    def test_qOpSizeMismatchWithNoiseModel(self):
        '''
        An exception gets thrown if the dimensions of the Kraus operators don't
        match the dimensions of the ``qOp`` when calling
        ``qOp.set_noise_model()``.
        '''

        twoQubitOperator = qOp().kron(qOp())
        with pytest.raises(WrongShapeError):
            twoQubitOperator.set_noise_model(damping_map(0.5))

    def test_kron_on_non_qOps(self):
        '''
        If `qOp.kron()` is called with any args not of type `qOp`,
        raise a TypeError.
        '''
        test_op = qOp()
        with pytest.raises(TypeError):
            test_op.kron(np.eye(2))
        with pytest.raises(TypeError):
            test_op.kron(test_op, np.eye(2))

    def test_qOpFailsWhenAppliedToDereferencedqReg(self):
        '''
        ``IllegalRegisterReference`` is raised when attempting to operate on a
        ``qReg``.
        '''
        test_op = qOp()
        q_reg = qReg()
        q_reg._qReg__is_dereferenced = True
        with pytest.raises(IllegalRegisterReference):
            test_op.on(q_reg)


class TestqOracleSuccess:
    def test_qOracle_is_subclass_of_qOp(self):
        '''
        Verifies that `qOracle` is a subclass of `qOp`.
        '''

        assert issubclass(qOracle, qOp)

    def test_qOracle_const_one_func(self):
        '''
        Verifies that the const function f(x) = 1
        yields the correct `qOracle`.
        '''

        blackBox = qOracle(lambda x: 1, 1)
        assert np.array_equal(
            blackBox._qOp__state.state(),
            [[0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]
        )

    def test_qOracle_const_zero_func(self):
        '''
        Verifies that the const function f(x) = 0
        yields the correct `qOracle`.
        '''

        blackBox = qOracle(lambda x: 0, 1)
        assert np.array_equal(
            blackBox._qOp__state.state(),
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )


class TestqOracleFailure:
    def test_classical_func_has_nonint_values(self):
        '''
        Checks that a `TypeError` is raise when the classical func arg
        to `qOracle` has non-int values in its range.
        '''
        with pytest.raises(TypeError):
            qOracle(lambda x: 0.1, 3)

    def test_classical_func_needs_to_be_callable(self):
        '''
        Raise a `TypeError` if the classical func used to create
        a `qOracle` isn't a callable.
        '''
        with pytest.raises(TypeError):
            qOracle(4, 5)

    def test_non_int_dimension_exponents(self):
        '''
        Checks that a `TypeError` gets raised when creating a
        `qOracle` with noninteger dimension exponents.
        '''
        with pytest.raises(TypeError):
            qOracle(lambda a: a, 0.1)
        with pytest.raises(TypeError):
            qOracle(lambda a: a, 1, 0.1)
        with pytest.raises(TypeError):
            qOracle(lambda a: a, 0.1, 0.1)
