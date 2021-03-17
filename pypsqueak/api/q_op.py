from functools import reduce
import numpy as np
from pypsqueak.errors import IllegalRegisterReference, WrongShapeError
from ._helpers import _convertQubitPermToHilbertPerm
from pypsqueak.noise import NoiseModel
from pypsqueak.squeakcore import Gate, Qubit


class qOp:
    '''
    A ``qOp`` is a high-level primitive which provides users with a
    representation of a quantum gate. In this implementation, the hardware of
    the gate is simulated with a ``pypsqueak.squeakcore.Gate`` object. Like the
    underlying ``Gate``, a ``qOp`` is a unitary operation. When instantiated
    with no arguments, the resulting ``qOp`` is the identity. Other operations
    can be represented by using a matrix representation of the operator as a
    creation argument. Additionally, noise can be modeled by providing a set of
    Kraus operators that characterizes said noise in the form of a
    pypsqueak.noise.NoiseModel object.

    Examples
    --------
    Here we demonstrate how to define the Pauli spin operators:

    >>> from pypsqueak.api import qOp
    >>> p_x = qOp([[0, 1], [1, 0]])
    >>> p_y = qOp([[0, -1j], [1j, 0]])
    >>> p_z = qOp([[1, 0], [0, -1]])

    The multiplication operator is overloaded to implement matrix
    multiplication:

    >>> p_x * p_x
    [[1 0]
     [0 1]]
    >>> p_y * p_y
    [[1 0]
     [0 1]]
    >>> iden = p_z * p_z
    >>> iden
    [[1 0]
     [0 1]]
    >>> p_x * p_y * p_z
    [[0.-1.j 0.+0.j]
     [0.+0.j 0.-1.j]]

    ``qOp`` instances are applied to ``qReg`` objects via the ``qOp.on()``
    method:

    >>> from pypsqueak.api import qReg
    >>> q = qReg()
    >>> q.peek()
    '(1.00e+00)|0>'
    >>> p_x.on(q)
    >>> q.peek()
    '(1.00e+00)|1>'

    We can define a function with return type ``qOp`` to implement
    parameterized gates:

    >>> import numpy as np
    >>> def rot_x(theta):
    ...     m_rep = [[np.cos(theta/2), -1j * np.sin(theta/2)],
    ...              [-1j * np.sin(theta/2), np.cos(theta/2)]]
    ...     return qOp(m_rep)
    ...
    >>> q = qReg()
    >>> rot_x(np.pi).on(q)
    >>> q.peek()
    '(6.12e-17+0.00e+00j)|0> + (0.00e+00-1.00e+00j)|1>'

    '''

    def __init__(self, matrix_rep=[[1, 0], [0, 1]], kraus_ops=None):
        self.__state = Gate(matrix_rep)
        self.set_noise_model(kraus_ops)

    def __noiseModelMatchesGateSize(self, kraus_ops):
        if kraus_ops is None:
            return True
        elif (not isinstance(kraus_ops, NoiseModel)
              or self.__state.shape() != kraus_ops.shape()):
            return False
        else:
            return True

    def set_noise_model(self, kraus_ops):
        '''
        Changes the NoiseModel on the ``qOp`` to that specified by
        ``kraus_ops``, Each of the elements of the ``NoiseModel``
        has the same dimensions (matching the ``qOp``) and they are
        collectively trace-preserving.

        By defualt ``kraus_ops = None``. The ``qOp`` is then noiselessly
        emulated. That this method would be absent/do nothing for a hardware
        implementation of the backend.

        Parameters
        ----------
        kraus_ops : NoiseModel or None
            A NoiseModel instance. Each element of the NoiseModel is an
            operation element in a generalized quantum operation. If ``None``,
            no noise is emulated.

        Examples
        --------
        If we want to model a noisy single-qubit channel, we can instantiate
        an identity operator with the corresponding noise. Let's make a
        channel exhibiting a bit flip noise with probability 0.5 of a flip,
        and then send a qubit in the |0> state through it 1000 times:

        >>> from pypsqueak.api import qReg, qOp
        >>> from pypsqueak.noise import b_flip_map
        >>> noisy_channel = qOp(kraus_ops=b_flip_map(0.5))
        >>> noisy_channel
        [[1 0]
         [0 1]]
        >>> send_results = []
        >>> for i in range(1000):
        ...      q = qReg()
        ...      noisy_channel.on(q)
        ...      send_results.append(q.measure(0))
        ...
        >>> n_zeros = 0
        >>> n_ones = 0
        >>> for result in send_results:
        ...     if result == 0:
        ...             n_zeros += 1
        ...     else:
        ...             n_ones += 1
        ...
        >>> n_zeros/1000
        0.512
        >>> n_ones/1000
        0.488

        To turn off noisy modeling, just call ``qOp.set_noise_model(None)``.

        '''
        if (not isinstance(kraus_ops, type(NoiseModel()))
                and kraus_ops is not None):
            raise TypeError("Noise model on a qOp must be of type "
                            "pypsqueak.noise.NoiseModel.")

        if not self.__noiseModelMatchesGateSize(kraus_ops):
            raise WrongShapeError("Size mismatch between Kraus "
                                  "operators and qOp.")
        self.__noise_model = kraus_ops

    def size(self):
        '''
        Returns the number of qubits that the ``qOp`` acts on. This is log base
        two of the dimensions of the corresponding matrix representation.

        Returns
        -------
        int
            The size of the ``qOp``.
        '''
        return len(self.__state)

    def shape(self):
        '''
        Returns the dimensions of the matrix representation of the ``qOp``.

        Returns
        -------
        tuple
            The shape of the matrix representation of the ``qOp``.
        '''

        return (2**len(self.__state),) * 2

    def on(self, q_reg, *targets):
        '''
        Applies a ``qOp`` to a ``qReg``. If the size of the ``qOp`` agrees with
        the size of the ``qReg``, no target qubits are required. If the ``qOp``
        is smaller than the ``qReg``, the ``qOp`` is lifted to the
        higher-dimensional Hilbert space of the ``qReg``. In that case, n
        target qubits must be, specified, where n is the size of the
        ``qOp`` before lifting. If the size of the ``qOp`` is larger than the
        size of the ``qReg``, an exception is raised.

        When the size of the ``qOp`` is smaller than the size of the ``qReg``,
        the ``targets`` specify how to order the qubits in the ``qReg``
        before application of the lifted operator (that is, the tensor product
        I^n (x) ``qOp``, where n is the length of the ``qReg`` minus the size
        of the ``qOp``). From left to right, the qubits named in the
        ``targets`` are swapped with the qubits at addresses zero, one, two,
        etc. All remaining qubits get bumped up to the next highest available
        register addresses which were NOT involved in the swap. After
        operation with the lifted ``qOp``, the ``qReg`` is permuted back to
        its original ordering.

        As an example, if a ``qReg`` is in the state |abcdef> and
        and ``qOp.on()`` is called with with ``targets`` = [3, 0, 4, 1], then
        the ``qReg`` is permuted to |adebfc> before application of the ``qOp``.

        If the size of the ``qOp`` and ``qReg`` match, then calling
        ``qOp.on()`` with no targets skips permutation of the register before
        applying the operator.

        Parameters
        ----------
        q_reg : pypsqueak.api.qReg
            The register to apply the operation to.
        *targets : int
            A list of locations in the register. The corresponding qubits are
            permuted to the lowest positions in the register before
            application of the operator. Must be nonnegative.

        Returns
        -------
        None
            This method only has the side effect of applying the ``qOp`` to a
            ``qReg``.

        Examples
        --------
        Here we apply a controlled NOT gate to the state |01> both with and
        without specifying targets:

        >>> from pypsqueak.api import qReg, qOp
        >>> from pypsqueak.gates import X, CNOT
        >>> q = qReg()
        >>> X.on(q)
        >>> q += 1
        >>> q.peek()
        '(1.00e+00)|01>'
        >>> CNOT.on(q)
        >>> q.peek()
        '(1.00e+00)|01>'
        >>> CNOT.on(q, 0, 1)
        '(1.00e+00)|01>'
        >>> CNOT.on(q, 1, 0)
        >>> q.peek()
        '(1.00e+00)|11>'

        Since the controlled NOT is a two-qubit gate, an exception is raised
        when we call it with only one target:

        >>> CNOT.on(q, 1)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "pypsqueak/api.py", line 763, in on
        pypsqueak.errors.WrongShapeError: Number of registers must match number
        of qubits gate operates on.
        '''

        self._validateRequestedGateApplication(q_reg, *targets)
        self._lift_register(q_reg, *targets)
        liftedGate = self._make_lifted_gate(q_reg)

        targetQubitSwapMatrix, targetQubitInverseSwap = (
            self.__generate_swap(q_reg, *targets))
        swappedQubitsReadyForGate = Qubit(
            np.dot(
                targetQubitSwapMatrix,
                q_reg._qReg__q_reg.state()))

        qubitsAfterGateApplication = self._applyGateToQubits(
            liftedGate,
            swappedQubitsReadyForGate)
        finalStateOfQubits = np.dot(
            targetQubitInverseSwap,
            qubitsAfterGateApplication.state())

        q_reg._qReg__q_reg.change_state(finalStateOfQubits)

    def _validateRequestedGateApplication(self, q_reg, *targets):
        if q_reg._qReg__is_dereferenced:
            raise IllegalRegisterReference(
                "Cannot operate on a dereferenced register.")

        self._validate_qOp_targets(*targets)
        self._validate_qOp_and_qReg_compatability(q_reg, *targets)

    def _validate_qOp_targets(self, *targets):
        if len(targets) != len(set(targets)):
            raise ValueError(
                'Specified quantum register targets for operation '
                'cannot contain duplicates.')

        if any([
                not isinstance(target, int) or target < 0
                for target in targets]):
            raise IndexError(
                        'Quantum register addresses must be nonnegative ints.')

    def _validate_qOp_and_qReg_compatability(self, q_reg, *targets):
        if len(targets) == 0:
            if self.size() != len(q_reg):
                raise IndexError(
                    'One or more targets must be specified for qOp to act on '
                    'when qOp and qReg are of different size.')
        else:
            if self.size() != len(targets):
                raise WrongShapeError(
                    'Number of target qubits must match the number of qubits '
                    'the qOp acts on.')

    def _lift_register(self, q_reg, *targets):
        '''
        Return a ``qReg`` with intermediate |0> qubits inserted
        after the last qubit if the largest target is out of bounds of the
        qReg. Return the same qReg if the largest target is within bounds.
        '''

        if len(targets) != 0 and max(targets) > len(q_reg) - 1:
            q_reg += max(targets) - len(q_reg) + 1

        return q_reg

    def _make_lifted_gate(self, q_reg):
        '''
        Return a ``Gate`` tensored on the left with the identity
        ``len(q_reg) - self.size()`` times. Returns an unraised ``Gate``
        if the register and operator are of the same size.
        '''

        if len(q_reg) == self.size():
            return self.__state
        else:
            left_identity_product = reduce(
                lambda a, b: a.gate_product(b),
                [Gate() for i in range(len(q_reg) - self.size())])
            liftedGate = left_identity_product.gate_product(self.__state)

        return liftedGate

    def _applyGateToQubits(self, gate, qubits):
        '''
        Applies a ``Gate`` to a ``Qubit`` object.
        '''

        qubitsAfterGateApplication = Qubit(
            np.dot(
                gate.state(),
                qubits.state()))

        if self.__noise_model is not None:
            self._simulateGateNoise(qubitsAfterGateApplication)

        return qubitsAfterGateApplication

    def _simulateGateNoise(self, qubits):
        '''
        Applies the NoiseModel on ``self`` to ``Qubit`` object.
        '''
        newStateEnsemble, probabilities = (
            self._makeNewStateEnsembleFromKrausOperators(qubits)
        )

        selected_state = np.random.choice(
            len(newStateEnsemble), p=probabilities)
        qubits.change_state(newStateEnsemble[selected_state])

    def _makeNewStateEnsembleFromKrausOperators(self, qubits):
        stateEnsemble = []
        probabilities = []

        for krausOp in self.__noise_model.getKrausOperators():
            # Raise each operator if necessary
            if len(qubits) > np.log2(krausOp.shape[0]):
                krausOp = np.kron(
                    np.identity(2**(len(qubits) - self.size())), krausOp)

            noise_result = krausOp.dot(qubits.state())
            noise_result_dual = np.conjugate(noise_result)
            probability = np.real(np.dot(noise_result, noise_result_dual))

            stateEnsemble.append(noise_result)
            probabilities.append(probability)

        return stateEnsemble, probabilities

    def __generate_swap(self, q_reg, *targets):
        '''
        Given a list of targets, generates matrix (and inverse) to swap targets
        into lowest qubit slot in register. Remaining qubits in register get
        bumped up, perserving order.

        Example: |abcdef> with targets = [3, 0, 4, 1] goes to |adebfc>
        '''

        # If no targets are specified, just return identity operators.
        if len(targets) == 0:
            return np.eye(2**len(q_reg)), np.eye(2**len(q_reg))

        self._validate_swap(q_reg, *targets)

        qubitOrderingAfterSwap = (list(targets)
                                  + [idx for idx in range(len(q_reg))
                                     if idx not in targets])
        qubitPermutationMatrix = np.array([
            [
                1.0 if (col == qubitOrderingAfterSwap[row]) else 0.0
                for col in range(len(q_reg))
            ]
            for row in range(len(q_reg))
        ])
        hilbertSpacePermutationMatrix = _convertQubitPermToHilbertPerm(
            q_reg, qubitPermutationMatrix)
        hilbertSpacePermutationInverse = hilbertSpacePermutationMatrix.T

        return hilbertSpacePermutationMatrix, hilbertSpacePermutationInverse

    def _validate_swap(self, q_reg, *targets):
        if len(q_reg) < max(targets):
            raise IndexError(
                "Uninitialized qubit referenced in swap operation.")

        if not all(isinstance(target, int) for target in targets):
            raise IndexError("Noninteger index encountered.")

        if any(target < 0 for target in targets):
            raise IndexError("Negative index encountered.")

    def dagger(self):
        '''
        Returns the Hermitian conjugate of the ``qOp``. This is equivalent to
        the transpose conjugate of the matrix representation.

        Returns
        -------
        pypsqueak.api.qOp
            The Hermitian conjugate of the operator.
        '''

        return qOp(self.__state.state().conj().T, kraus_ops=self.__noise_model)

    def __mul__(self, another_op):
        '''
        Returns the matrix product (another_op)(some_op) i.e. with another_op
        acting second. The resulting noise model is that of ``some_op``.
        '''
        if self.size() != another_op.size():
            raise WrongShapeError("qOp size mismatch.")

        return qOp(
            np.dot(another_op._qOp__state.state(), self.__state.state()),
            kraus_ops=self.__noise_model)

    def kron(self, another_op, *more_ops):
        '''
        Returns the tensor product (implemented as a matrix Kronecker product)
        of ``self`` (x) ``another_op``. Optionally continues to tensor-in
        additional ops in ``more_ops``. Ignores noise model set on any of the
        factors.

        Parameters
        ----------
        another_op : pypsqueak.api.qOp
            Right-hand factor in Kronecker product.
        *more_ops : pypsqueak.api.qOp
            Additional optional factors in Kronecker product.

        Returns
        -------
        pypsqueak.api.qOp
            Kronecker product.

        Examples
        --------
        Here we build the identity operator acting on three qubits:

        >>> from pypsqueak.api import qOp
        >>> iden_3 = qOp().kron(qOp(), qOp())
        >>> iden_3
        [[1 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0]
         [0 0 0 1 0 0 0 0]
         [0 0 0 0 1 0 0 0]
         [0 0 0 0 0 1 0 0]
         [0 0 0 0 0 0 1 0]
         [0 0 0 0 0 0 0 1]]

        Less trivially, let's make the operator applying an X gate to the first
        qubit and the identity operator to the zeroeth qubit:

        >>> from pypsqueak.gates import X
        >>> X.kron(qOp())
        [[0 0 1 0]
         [0 0 0 1]
         [1 0 0 0]
         [0 1 0 0]]

        '''
        listOfArgQOps = [another_op] + list(more_ops)
        for op in listOfArgQOps:
            if not isinstance(op, qOp):
                raise TypeError("Arguments must be qOp objects.")

        kroneckerProduct = self.__state.gate_product(
            *[op._qOp__state for op in listOfArgQOps]).state()
        return qOp(kroneckerProduct)

    def __repr__(self):
        return str(self.__state)
