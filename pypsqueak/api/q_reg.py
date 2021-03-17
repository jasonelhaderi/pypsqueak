import numpy as np
from .q_op import qOp
from pypsqueak.errors import (IllegalRegisterReference, WrongShapeError,
                              NonUnitaryInputError, IllegalCopyAttempt)
from pypsqueak.squeakcore import Qubit
from ._helpers import _makeProjectorOnToSubspace
from pypsqueak.squeakcore._helpers import _is_unitary


class qReg:
    '''
    A ``qReg`` is a high-level primitive which provides users with a
    representation of a quantum register. In this implementation, the quantum
    device on which the register exists is simulated via a
    ``pypsqueak.squeakcore.Qubit`` object. Like the underlying ``Qubit``, a
    ``qReg`` is initialized in the |0> state.

    Per the no-cloning theorem, any attempt to copy a ``qReg`` object will
    throw an exception. Additionally, operations which would otherwise leave
    duplicates of a specific instance of a ``qReg`` lying around 'dereference'
    the register. Once a ``qReg`` is dereferenced, any attempt to interact with
    the ``qReg`` will throw an exception.

    Since this implementation uses simulated quantum hardware, methods for
    examining the quantum state as a Dirac ket and returning the state as a
    numpy array are provided. They are ``qReg.peek()`` and
    ``qReg.dump_state()``, respectively.

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

    Note that different instances of a ``qReg`` are considered unequal even if
    the underlying state is the same. Additionally, when ``qOp.on()`` is
    applied to a target in a ``qReg`` that is outside the range of the
    register, new filler qubits are automatically initialzed in the zero state.

    Now we demonstrate which operators are overloaded for ``qReg`` objects as
    well as their behavior. We can append any number of qubits to a ``qReg``
    like so:

    >>> from pypsqueak.gates import X
    >>> a = qReg(1)
    >>> X.on(a, 1)
    >>> a += 3
    >>> a.peek()
    '(1.00e+00)|00010>'

    Two registers can be joined into one via the tensor product. This can be
    done in place:

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

    pypsqueak.errors.IllegalRegisterReference: Dereferenced register
    encountered.

    Notice that taking the product of ``qReg`` objects dereferences any
    operands.
    '''

    def __init__(self, n_qubits=1):
        if n_qubits < 1:
            raise ValueError("A qReg must have length of at least 1.")

        init_state = [0 for i in range(2**n_qubits)]
        init_state[0] = 1
        self.__q_reg = Qubit(init_state)
        self.__is_dereferenced = False

    def measure(self, target):
        '''
        Performs a projective measurement on the qubit at the address
        ``target``. In this simulated implementation, there are three steps:

        #. Compute probability of each measurement using the amplitudes of each
           basis vector in the computational basis decomposition.
        #. Use these probabilities to randomly pick a measurement result.
        #. Project the ``qReg`` state onto the result's corresponding
           eigenspace.

        Parameters
        ----------
        target : int
            The index of the qubit in the ``qReg`` to measure. An exception
            gets
            thrown if the value is negative or out of range.

        Returns
        -------
        int
            Either a one or a zero, depending on the result of the measurement.
            The state of the ``qReg`` is projected onto the corresponding
            subspace.

        Examples
        --------
        Here we prepare the Bell state and then measure qubit one in the
        ``qReg``.

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

        self._throwExceptionIfRequestedMeasurementIsNotValid(target)

        measurement_outcome = self.measure_observable(
            self._makeComputationalBasisObservable(target))

        # Two-state observable eigenvalues are 1 and -1,
        # so manual translation to 0 and 1 (respectively) is needed.
        return 0 if measurement_outcome == 1 else 1

    def measure_observable(self, observable):
        '''
        Performs a projective measurement of the ``observable`` corresponding
        to a ``qOp``.

        Parameters
        ----------
        observable : pypsqueak.api.qOp
            The ``qOp`` corresponding to some observable to measure. An
            exception gets thrown if its size larger than the size of the
            ``qReg``.

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

        self._throwExceptionIfObservableMeasurementIsNotValid(observable)
        observable = self._liftOperatorToDimensionOfRegister(observable)

        measurementEigenvalues, measurementTransitionMatrix = np.linalg.eig(
            observable._qOp__state.state())
        measurementTransitionProbabilities = (
            self._generateStateTransitionProbabilities(
                measurementTransitionMatrix)
        )

        measurementResult = np.random.choice(
            measurementEigenvalues,
            p=measurementTransitionProbabilities)

        self._collapseRegisterWavefunction(
            measurementResult,
            measurementEigenvalues,
            measurementTransitionMatrix)

        return measurementResult

    def peek(self):
        '''
        Returns a ket description of the state of a ``qReg``. Would have no
        effect on hardware implementations of the backend. If the register has
        been dereferenced, raises an exception.

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

        After dereferencing the register via a multiplication, calling
        ``peek()`` raises an exception:

        >>> a * qReg()
        qReg(4)
        >>> a.peek()
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "pypsqueak/api.py", line 309, in peek
                raise IllegalRegisterReference('Dereferenced register '
                                                     'encountered.')
        pypsqueak.errors.IllegalRegisterReference: Dereferenced register
        encountered.

        '''

        if self.__is_dereferenced:
            raise IllegalRegisterReference('Dereferenced register '
                                           'encountered.')

        return str(self.__q_reg)

    def dump_state(self):
        '''
        Returns a copy of the state of a ``qReg`` as a numpy array. Would have
        no effect on a hardware implementation of the backend. If the register
        has been dereferenced, raises an exception.

        Returns
        -------
        numpy.ndarray
            The state of ``qReg`` as a vector in the computational basis. Has
            no side effects.

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

        Now we dereference the ``qReg`` and run into an exception when we try
        to dump its state again:

        >>> a * qReg()
        qReg(4)
        >>> a.dump_state()
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "pypsqueak/api.py", line 342, in dump_state
            exception.
        pypsqueak.errors.IllegalRegisterReference: Dereferenced register
        encountered.

        '''

        if self.__is_dereferenced:
            raise IllegalRegisterReference('Dereferenced register '
                                           'encountered.')

        return self.__q_reg.state()

    def _throwExceptionIfRequestedMeasurementIsNotValid(self, target):
        if self.__is_dereferenced:
            raise IllegalRegisterReference('Measurement attempted on '
                                           'dereferenced register.')

        isTargetQubitIndexValid = (isinstance(target, int)
                                   and target >= 0
                                   and target < len(self))
        if not isTargetQubitIndexValid:
            raise IndexError('Quantum register address must be nonnegative '
                             'integer less than size of register.')

    def _throwExceptionIfObservableMeasurementIsNotValid(self, observable):
        if self.__is_dereferenced:
            raise IllegalRegisterReference("Measurement attempted on "
                                           "dereferenced register.")

        if not isinstance(observable, qOp):
            raise TypeError("Argument of measure_observable() must be a qOp.")

        if len(self) < observable.size():
            raise WrongShapeError("Observable larger than qReg.")

    def _liftOperatorToDimensionOfRegister(self, operator):
        '''
        Takes the ``qOp`` ``operator`` and if its size is smaller than the
        ``qReg``, returns a 'lifted' version of the operator that has been
        prepended with the tensor product of
        ``len(qReg) - operator.size()`` copies of the identity operator. If
        the size of ``operator`` already matches that of the ``qReg``, an
        unchanged version of ``operator`` is returned.

        Parameters
        ----------
        operator : qOp
        The operator to be lifted.

        Returns
        -------
        qOp
        A version of the operator lifted to be of the same dimension as the
        qReg.
        '''
        if len(self) > operator.size():
            diff = len(self) - operator.size()
            iden = qOp(np.eye(2**diff))
            operator = iden.kron(operator)

        return operator

    def _makeComputationalBasisObservable(self, target):
        '''
        Returns a ``qOp`` corresponding to the observable for a computational
        basis measurement on the qubit indexed by ``target``. Since this takes
        the form (identity operator)^m * (Pauli Z) * (identity operator)^m,
        with m + n + 1 equal the size of the ``qReg``, the eigenvalues of this
        operator are 1 and -1.
        '''
        from pypsqueak.gates import I, Z

        if target == len(self) - 1:
            observable = Z
        else:
            observable = I

        for i in reversed(range(len(self) - 1)):
            if i == target:
                observable = observable.kron(Z)
            else:
                observable = observable.kron(I)

        return observable

    def _generateStateTransitionProbabilities(self, transitionMatrix):
        '''
        Returns a list of the transition probabilities from the current
        ``qReg`` state to the set of normalized states specified by each column
        of the array ``transitionMatrix``.

        Parameters
        ----------
        potentialTransitions : array
        A list of normalized eigenvectors representing possible new states for
        the ``qReg``.

        Returns
        -------
        list
        A list of corresponding transition probabilities for each possible new
        state.
        '''

        if not _is_unitary(transitionMatrix):
            raise NonUnitaryInputError("Non-unitary transition matrix "
                                       "encountered while computing "
                                       "qReg transition probabilities.")

        currentStateAsRowVector = self.__q_reg.state().T
        transitionAmplitudes = np.dot(
            currentStateAsRowVector,
            transitionMatrix
        )
        transitionProbabilities = [
            amplitude * amplitude.conj() for amplitude in transitionAmplitudes
        ]

        return transitionProbabilities

    def _collapseRegisterWavefunction(
            self,
            measurementResult,
            measurementEigenvalues,
            measurementTransitionMatrix):
        '''
        Collapses the ``qReg`` to the state corresponding to a measurement of
        ``measurementResult`` for an observable with eigenvalues given by the
        list ``measurementEigenvalues`` and normalized measurement outcomes
        given by the columns of ``measurementTransitionMatrix``.
        '''

        collapsedState = np.dot(
            _makeProjectorOnToSubspace(
                measurementResult,
                measurementEigenvalues,
                measurementTransitionMatrix),
            self.__q_reg.state())

        self.__q_reg.change_state(collapsedState)

    def __iadd__(self, n_new_qubits):
        '''
        Prepends ``n_new_qubits`` qubits to the register in the |0> state.
        Leaves register unchanged if n_new_qubits is zero.
        '''

        if self.__is_dereferenced:
            raise IllegalRegisterReference('Attempt to add Qubits to '
                                           'dereferenced register.')

        if not isinstance(n_new_qubits, int) or n_new_qubits < 0:
            raise ValueError("Can only add a nonnegative integer number of "
                             "qubits to qReg.")

        n_qubits_in_zero_state = Qubit(
            [1 if i == 0 else 0 for i in range(2**n_new_qubits)])

        self.__q_reg = n_qubits_in_zero_state.qubit_product(self.__q_reg)
        return self

    def __imul__(self, some_reg):
        '''
        Concatentates the register with some_reg (|a_reg> *= |some_reg> stores
        |a_reg>|some_reg> into ``a_reg``).
        '''

        if not isinstance(some_reg, qReg):
            raise TypeError("Cannot concatentate a non-qReg to a qReg.")

        if self.__is_dereferenced or some_reg.__is_dereferenced:
            raise IllegalRegisterReference(
                'Concatentation attempted on dereferenced register.')

        self.__q_reg = self.__q_reg.qubit_product(some_reg._qReg__q_reg)
        some_reg._qReg__is_dereferenced = True

        return self

    def __mul__(self, another_reg):
        '''
        For concatenating the register with another_reg
        (|new> = |reg> * |another_reg> stores the product into ``new``).
        '''

        if self.__is_dereferenced or another_reg.__is_dereferenced:
            raise IllegalRegisterReference(
                'Concatenation attempted on dereferenced register.')

        product_register = qReg()
        product_qubits = self.__q_reg.qubit_product(another_reg._qReg__q_reg)
        product_register._qReg__q_reg.change_state(product_qubits.state())

        self.__is_dereferenced = True
        another_reg._qReg__is_dereferenced = True

        return product_register

    def __len__(self):
        if self.__is_dereferenced:
            raise IllegalRegisterReference(
                'Dereferenced register encountered.')

        return len(self.__q_reg)

    def __copy__(self):
        raise IllegalCopyAttempt('Cannot copy a qReg.')

    def __deepcopy__(self, memo):
        raise IllegalCopyAttempt('Cannot copy a qReg.')

    def __repr__(self):
        if not self.__is_dereferenced:
            return "qReg({})".format(len(self))
        else:
            return "Dereferenced qReg"
