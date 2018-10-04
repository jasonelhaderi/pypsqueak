'''
Front end of pypSQUEAK. Contains definitions of ``qReg``, ``qOp``, and ``qOracle``
classes which provide abstract representations of quantum hardware.
'''
import numpy as np
import copy
import cmath

from pypsqueak.squeakcore import Qubit, Gate
import pypsqueak.errors as sqerr

class qReg:
    '''
    A high-level primitive which provides users with a means of interfacing with
    a quantum device (simulated in this implementation).

    A ``qReg`` is a high-level primitive which provides users with a representation
    of a quantum register. In this implementation, the quantum device on which
    the register exists is simulated via a ``pypsqueak.squeakcore.Qubit`` object.
    Like the underlying ``Qubit``, a ``qReg`` is initialized in the |0> state. This can be
    overridden if the ``qReg`` is instead instantiated with some other numeric vector
    as argument (the resulting ``qReg`` will use a normalized version of
    that vector).

    As per the no-cloning theorem, any attempt to copy a ``qReg`` object will
    throw an exception. Additionally, operations which would otherwise leave
    duplicates of a specific instance of a ``qReg`` lying around dereference the
    register. Once a ``qReg`` is dereferenced, any attempt to interact with the
    ``qReg`` will throw an exception.

    Since this implementation uses simulated quantum hardware, methods for examining
    the quantum state as a Dirac ket and returning the state as a numpy array are
    provided. They are ``qReg.peek()`` and ``qReg.dump_state()``, respectively.

    Examples
    --------
    Here we demonstrate three ways to initialize a ``qReg`` with 3 qubits.

    >>> from pypsqueak.api import qReg, qOp
    >>> a = qReg(3)
    >>> b = qReg()
    >>> b += 2
    >>> c = qReg()
    >>> identity_op = qOp()
    >>> identity_op.on(c, 2)
    >>> a == b
    False
    >>> a.dump_state() == b.dump_state()
    array([ True,  True,  True,  True,  True,  True,  True,  True])
    >>> a.dump_state() == c.dump_state()
    array([ True,  True,  True,  True,  True,  True,  True,  True])
    >>> a.peek()
    '(1.00e+00)|000>'

    A couple of quick notes are in order. Observe that new ``qReg`` instances are
    initialized to the zero state of the computational basis. Additionally,
    different instances of a ``qReg`` are considered unequal even if the
    underlying state is the same. Lastly, when ``qOp.on()`` is applied to a target
    in a ``qReg`` that is outside the range of the register, new filler qubits
    are automatically initialzed in the zero state.

    Now we demonstrate which operators are overloaded for ``qReg`` objects as well
    as their behavior. We can append any number of qubits to a ``qReg`` like so:

    >>> from pypsqueak.gates import X
    >>> a = qReg(1)
    >>> X.on(a, 1)
    >>> a += 3
    >>> a.peek()
    '(1.00e+00)|00010>'

    Two registers can be joined into one via the tensor product. This can be done
    in place:

    >>> a *= qReg(2)
    >>> a.peek()
    '(1.00e+00)|0001000>'

    A new ``qReg`` can be created similarly:

    >>> a = qReg()
    >>> X.on(a, 0)
    >>> b = qReg()
    >>> c = a * b
    >>> c
    qReg(2)
    >>> c.peek()
    '(1.00e+00)|10>'
    >>> a
    Dereferenced qReg
    >>> a.peek()
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "pypsqueak/api.py", line 199, in peek

    pypsqueak.errors.IllegalRegisterReference: Dereferenced register encountered.

    Notice that taking the product of ``qReg`` objects dereferences any operands.

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
        In this simulated implementation, there are three steps:

        #. Compute probability of each measurement using the amplitudes of each
           basis vector in the computational basis decomposition.
        #. Use these probabilities to randomly pick a measurement result.
        #. Project onto the result's corresponding eigenspace.

        Parameters
        ----------
        target : int
            The index of the qubit in the ``qReg`` to measure. An exception gets
            thrown if the value is negative or out of range.

        Returns
        -------
        int
            Either a one or a zero, depending on the result of the measurement.
            The state of the ``qReg`` is projected onto the corresponding subspace.

        Examples
        --------
        Here we prepare the Bell state and then measure qubit one in the ``qReg``.

        >>> from pypsqueak.api import qReg
        >>> from pypsqueak.gates import CNOT, H
        >>> a = qReg()
        >>> H.on(a, 0)
        >>> CNOT.on(a, 1, 0)
        >>> a.peek()
        '(7.07e-01)|00> + (7.07e-01)|11>'
        >>> a.measure(1)
        1
        >>> a.peek()
        '(1.00e+00)|11>'

        '''

        if self.__killed:
            raise sqerr.IllegalRegisterReference('Measurement attempted on dereferenced register.')

        if not isinstance(target, int) or target < 0:
            raise IndexError('Quantum register address must be nonnegative integer.')

        if target > len(self) - 1:
            raise IndexError('Specified quantum register address out of range.')

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
        Performs a projective measurement of the ``observable`` corresponding to a
        ``qOp``. In this simulated implementation, four steps are involved:

        #. Determine measurement outcomes.
        #. Compute probability of each measurement using the amplitudes of each
           basis vector in the computational basis decomposition.
        #. Use these probabilities to randomly pick a measurement result.
        #. Project onto the result's corresponding eigenspace.

        Parameters
        ----------
        observable : pypsqueak.api.qOp
            The ``qOp`` corresponding to some observable to measure. An exception gets
            thrown if its size larger than the size of the ``qReg``.

        Returns
        -------
        int
            One of the eigenvalues of ``observable``. The state of the ``qReg``
            is projected onto the corresponding subspace.

        Examples
        --------
        When the size of the operator is smaller than the ``qReg``, the
        the operator is prepended with identity operators.

        >>> from pypsqueak.api import qReg, qOp
        >>> from pypsqueak.gates import X
        >>> a = qReg(3)
        >>> X.on(a, 0)
        >>> a.peek()
        '(1.00e+00)|001>'
        >>> print(a.measure_observable(X))
        -1.0
        >>> a.peek()
        '(-7.07e-01)|000> + (7.07e-01)|001>'

        '''

        if not isinstance(observable, type(qOp())):
            raise TypeError("Argument of measure_observable() must be a qOp.")

        if len(self) < observable.size():
            raise sqerr.WrongShapeError("Observable larger than qReg.")

        if len(self) > observable.size():
            diff = len(self) - observable.size()
            iden = qOp(np.eye(2**diff))
            observable = iden.kron(observable)

        # Determine normalized eigenvalue/vector pairs.
        e_vals, e_vecs = np.linalg.eig(observable._qOp__state.state())

        # Compute probabilities
        probabilities = []
        current_state = self.__q_reg.state()
        for i in range(len(e_vecs)):
            amplitude = np.dot(current_state, e_vecs[:, i])
            probabilities.append(amplitude * amplitude.conj())

        # Choose measurement result
        measurement_index = np.random.choice([i for i in range(len(e_vals))], p=probabilities)
        measurement_result = e_vals[measurement_index]

        # Build subspace corresponding to eigenvalue
        subspace_basis = []
        for i in range(len(e_vals)):
            if e_vals[i] == measurement_result:
                subspace_basis.append(e_vecs[:,i])

        # Make projection operator
        projector = np.outer(subspace_basis[0], subspace_basis[0])
        for i in range(1, len(subspace_basis)):
            projector += np.outer(subspace_basis[i], subspace_basis[i])

        new_state = np.dot(projector, current_state)
        self.__q_reg.change_state(new_state)

        return measurement_result

    def peek(self):
        '''
        Returns a ket description of the state of a ``qReg``. Note that this
        is impossible on hardware implementations as a consequence of the
        no-cloning theorem. If the register has been dereferenced, raises an
        exception.

        Returns
        -------
        str
            Description of ``qReg`` state. Has no side effects.

        Examples
        --------
        Here we peek at a register in the Hadamard state:

        >>> from pypsqueak.api import qReg
        >>> from pypsqueak.gates import H
        >>> a = qReg(3)
        >>> H.on(a, 0)
        >>> a.peek()
        '(7.07e-01)|000> + (7.07e-01)|001>'

        After dereferencing the register via a multiplication, calling ``peek()`` raises an
        exception:

        >>> a * qReg()
        qReg(4)
        >>> a.peek()
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "pypsqueak/api.py", line 309, in peek
                raise sqerr.IllegalRegisterReference('Dereferenced register encountered.')
        pypsqueak.errors.IllegalRegisterReference: Dereferenced register encountered.

        '''

        if self.__killed:
            raise sqerr.IllegalRegisterReference('Dereferenced register encountered.')

        return str(self.__q_reg)

    def dump_state(self):

        '''
        Returns a copy of the state of a ``qReg`` as a numpy array. Note that this
        is impossible on hardware implementations as a consequence of the
        no-cloning theorem. If the register has been dereferenced, raises an
        exception.

        Returns
        -------
        numpy.ndarray
            The state of ``qReg`` as a vector in the computational basis. Has no side effects.

        Examples
        --------
        Here we get a vector corresponding to the Hadamard state:

        >>> from pypsqueak.api import qReg
        >>> from pypsqueak.gates import H
        >>> a = qReg(3)
        >>> H.on(a, 0)
        >>> a.dump_state()
        array([0.70710678, 0.70710678, 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        ])

        Now we dereference the ``qReg`` and run into an exception when we try to
        dump its state again:

        >>> a * qReg()
        qReg(4)
        >>> a.dump_state()
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "pypsqueak/api.py", line 342, in dump_state
            exception.
        pypsqueak.errors.IllegalRegisterReference: Dereferenced register encountered.

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

class qOp:

    '''
    A high-level primitive for representing unitary gates. In this implementation,
    noise can be simulated by instantiating a ``qOp`` with the kwarg ``kraus_ops``,
    a list of operation elements characterizing a noisy quantum operation.
    A high-level primitive which provides users with a means of interfacing with
    a quantum device (simulated in this implementation).

    A ``qOp`` is a high-level primitive which provides users with a representation
    of a quantum gate. In this implementation, the hardware of the gate is simulated
    with a ``pypsqueak.squeakcore.Gate`` object. Like the underlying
    ``Gate``, a ``qOp`` is by default a unitary operation. When instantiated
    with no aguments, the resulting ``qOp`` is the identity. Other operations
    can be represented by using a matrix representation of the operator as a
    creation argument. Additionally, noise can be modeled by providing a list of
    the the operation elements (as ``numpy.ndarray``s) characterizing said noise.

    Examples
    --------
    Here we demonstrate how to define the Pauli spin operators:

    >>> from pypsqueak.api import qOp
    >>> p_x = qOp([[0, 1], [1, 0]])
    >>> p_y = qOp([[0, -1j], [1j, 0]])
    >>> p_z = qOp([[1, 0], [0, -1]])

    The multiplication operator is overloaded to implement matrix multiplication:

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

    ``qOp``s are applied to ``qReg`` objects via the ``qOp.on()`` method:

    >>> from pypsqueak.api import qReg
    >>> q = qReg()
    >>> q.peek()
    '(1.00e+00)|0>'
    >>> p_x.on(q)
    >>> q.peek()
    '(1.00e+00)|1>'

    We can define a function with return type ``qOp`` to implement parameterized
    gates:

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
        self.__validate_matrix_rep(matrix_rep)
        self.__state = Gate(matrix_rep)

        self.__validate_kraus_ops(kraus_ops)
        self.__noise_model = kraus_ops

    def __validate_matrix_rep(self, matrix_rep):
        # Checks that input is list-like, square matrix
        try:
            n_cols = len(matrix_rep)
            n_rows = len(matrix_rep[0])
            if n_rows != n_cols:
                raise TypeError('Input must be a square matrix.')
            for row in matrix_rep:
                if len(row) != n_cols:
                    raise TypeError('Input must be a square matrix.')
                for element in row:
                    try:
                        element + 5
                    except:
                        raise TypeError('Elements of matrix_rep must be numeric.')
        except:
            raise TypeError('Input must be matrix-like.')

        # Checks that the input has valid dimensions
        if not sqerr._is_power_2(len(matrix_rep)) or len(matrix_rep) < 2:
            raise sqerr.WrongShapeError('matrix_rep must be nXn with n > 1 a power of 2.')

        # Checks that the input is unitary
        product_with_conj = np.dot(np.asarray(matrix_rep).T.conj(), matrix_rep)
        is_unitary = np.allclose(product_with_conj, np.eye(len(matrix_rep)))
        if is_unitary == False:
            raise sqerr.NonUnitaryInputError('matrix_rep must be unitary.')


    def __validate_kraus_ops(self, kraus_ops):
        # Checks on the validity of the Kraus operators
        # Check that the size of the Kraus operators agrees with the gate size
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

            # Check that kraus_ops are trace-preserving
            identity = np.identity(kraus_shape[0])
            sum = np.matmul(np.conjugate(kraus_ops[0].T), kraus_ops[0])
            for i in range(1, len(kraus_ops)):
                sum += np.matmul(np.conjugate(kraus_ops[i].T), kraus_ops[i])

            if not np.allclose(sum, identity):
                raise sqerr.NormalizationError("Kraus operators must be trace-preserving.")

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
            gate_shape = self.__state.shape()
            if gate_shape != kraus_shape:
                raise sqerr.WrongShapeError("Size mismatch between Kraus operators and gate.")

    def set_noise_model(self, kraus_ops):
        '''
        Changes the noise model on the ``qOp`` to that specified by the list
        of numpy ndarrays ``kraus_ops``. Each element in this list must have the
        same dimensions, match the size of the ``qOp``, and collectively be
        trace-preserving.

        By defualt ``kraus_ops = None``. The ``qOp`` is then noiselessly emulated. Note that
        this method would be absent from a hardware implementation of SQUEAK.

        Parameters
        ----------
        kraus_ops : list or None
            A list of numpy ndarrays. Each element of the list is an operation
            element in a generalized quantum operation. If ``None``, then no
            noise is emulated.

        Examples
        --------
        If we want to model a noisy single-qubit channel, we can instantiate an identity
        operator with the corresponding noise. Let's make a channel exhibiting a bit
        flip noise with probability 0.5 of a flip, and then send a qubit in the |0>
        state through it 1000 times:

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

        self.__validate_kraus_ops(kraus_ops)
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
        is smaller than the ``qReg``, the ``qOp`` is lifted to the higher-dimensional
        Hilbert space of the ``qReg``. In that case, n target qubits must be,
        specified, where n is the size of the ``qOp`` before lifting. If the
        size of the ``qOp`` is larger than the size of the ``qReg``, an
        exception is raised.

        When the size of the ``qOp`` is smaller than the size of the ``qReg``,
        the ``targets`` specify how to order the qubits in the ``qReg``
        before application of the lifted operator (that is, the tensor product
        I^n (x) ``qOp``, where n is the length of the ``qReg`` minus the size of
        the ``qOp``). From left to right, the qubits named in the ``targets``
        are swapped with the qubits at addresses zero, one, two, etc. All remaining
        qubits get bumped up to the next highest available register addresses which
        were NOT involved in the swap. After operation with the lifted ``qOp``,
        the ``qReg`` is permuted back to its original ordering.

        As an example, if a ``qReg`` is in the state |abcdef> and
        and ``qOp.on()`` is called with with ``targets`` = [3, 0, 4, 1], then
        the ``qReg`` is permuted to |adebfc> before application of the ``qOp``.

        If the size of the ``qOp`` and ``qReg`` match, then calling ``qOp.on()``
        with no targets skips permutation of the register before applying the
        operator.

        Parameters
        ----------
        q_reg : pypsqueak.api.qReg
            The register to apply the operation to.
        *targets : int
            A list of locations in the register. The corresponding qubits are
            permuted to the lowest positions in the register before application of
            the operator. Must be nonnegative.

        Returns
        -------
        None
            This method only has the side effect of applying the ``qOp`` to a
            ``qReg``.

        Examples
        --------
        Here we apply a controlled NOT gate to the state |01> both with and without
        specifying targets:

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

        Since the controlled NOT is a two-qubit gate, an exception is raised when
        we call it with only one target:

        >>> CNOT.on(q, 1)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "pypsqueak/api.py", line 763, in on
                # Check that the gate size matches the number of quantum_reg locations
        pypsqueak.errors.WrongShapeError: Number of registers must match number of qubits gate operates on.

        '''

        if q_reg._qReg__killed:
            raise sqerr.IllegalRegisterReference("Cannot operate on a dereferenced register.")

        # Check that at least one quantum_reg location is specified when gate and register
        # sizes don't match.
        if len(targets) == 0 and self.size() != len(q_reg):
            raise IndexError('One or more targets must be specified for gate and register of different size.')

        # Check that there are no duplicate register locations for the instruction
        if len(targets) != len(set(targets)):
            raise ValueError('Specified quantum register targets must be unique.')

        # Check that the gate size matches the number of quantum_reg locations
        if len(targets) != 0 and self.size() != len(targets):
            raise sqerr.WrongShapeError('Number of registers must match number of qubits gate operates on.')

        if len(targets) != 0:
            # Check that all the register locations are nonnegative integers
            for address in targets:
                if not isinstance(address, int):
                    raise IndexError('Quantum register addresses must be integer.')

                if address < 0:
                    raise IndexError('Quantum register addresses must be nonnegative.')

            # If any of the specified quantum_reg addresses have not yet been initialized,
            # initialize them (as well as intermediate reg locs) in the |0> state
            if max(targets) > len(q_reg) - 1:
                q_reg += max(targets) - len(q_reg) + 1

        # Initialize an identity gate for later use.
        iden = Gate()

        # If no targets are specified, __generate_swap() returns identity matricies.
        swap, swap_inverse = self.__generate_swap(q_reg, *targets)

        before_swap_and_op = q_reg._qReg__q_reg.state()
        after_swap_before_op = np.dot(swap, before_swap_and_op)

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
            after_swap_after_op = np.dot(operator.state(), after_swap_before_op)

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

        new_reg_state = np.dot(swap_inverse, after_swap_after_op)
        q_reg._qReg__q_reg.change_state(new_reg_state)

    def __generate_swap(self, q_reg, *targets):
        '''
        Given a list of targets, generates matrix (and inverse) to swap targets
        into lowest qubit slot in register. Remaining qubits in register get
        bumped up, perserving order.

        Example: |abcdef> with targets = [3, 0, 4, 1] goes to |adebfc>
        '''

        # If no targets are specified, just return identity operators
        # for the permutation matric and its inverse.
        if len(targets) == 0:
            return np.eye(2**len(q_reg)), np.eye(2**len(q_reg))

        # Check that the targets are valid.
        if len(q_reg) < max(targets):
            raise IndexError("Uninitialized qubit referenced in swap operation.")

        if not all(isinstance(target, int) for target in targets):
            raise IndexError("Noninteger index encountered.")

        if any(target < 0 for target in targets):
            raise IndexError("Negative index encountered.")

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

    def dagger(self):
        '''
        Returns the Hermitian conjugate of the ``qOp``. This is equivalent to
        the transpose conjugate of the matrix representation.

        Returns
        -------
        pypsqueak.api.qOp
            The Hermitian conjugate of the operator.
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
        Returns the tensor product (implemented as a matrix Kronecker product)
        of ``self`` (x) ``another_op``. Optionally continues to tensor-in
        additional ops in ``more_ops``.

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
    Subclass of ``qOp``. Useful for representing quantum black-boxes, such as
    that appearing in the Deutsch-Jozsa algorithm.

    ``qOracle`` implements a unitary transformation U_f|x>|y> = |x>|y XOR f(x)>
    where the classical function f maps nonnegative integers to nonnegative
    integers. Note that XOR is performed bitwise on the computational basis
    label y and f(x). This reduces to mod 2 addition when y and f(x) are both
    one bit long. ``n`` specifies the number of qubits in the left side portion
    of the quantum register, while ``m`` specifies the number of qubits in the
    right side portion.
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
        Raises an exception if the argument isn't nonnegative or if larger than
        the ``n`` portion of the intended register.

        Parameters
        ----------
        x_val : int
            Argument of function.

        Returns
        -------
        int
            Value of the function.

        Examples
        --------

        >>> from pypsqueak.api import qOracle
        >>> black_box = qOracle(lambda x: 0, 2)
        >>> black_box.classical_func(1)
        0
        >> black_box.classical_func(3)
        0
        >> black_box.classical_func(3)
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "pypsqueak/api.py", line 1065, in classical_func
            raise ValueError("Classical function input out of bounds.")
        ValueError: Classical function input out of bounds.

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
