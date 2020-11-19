'''
Implements functions returning sets of trace-one Kraus operators. Each function
corresponds to a specific kind of one-qubit noise. For an example of usage, see
:func:`~pypsqueak.api.qOp.set_noise_model`.
'''

import numpy as np
from pypsqueak.squeakcore import _is_numeric_square_matrix
from pypsqueak.errors import NormalizationError, WrongShapeError
from functools import reduce


class NoiseModel:
    '''
    A map characterizing a kind of noise to be simulated in a `qOp`.
    '''

    def __init__(self, kraus_ops=None):
        if kraus_ops is None:
            self._krausOperators = []
            self._shape = None
        else:
            self.setKrausOperators(kraus_ops)

    def getKrausOperators(self):
        return self._krausOperators

    def setKrausOperators(self, listOfKrausOperators):
        _validateListOfKrausOperators(listOfKrausOperators)

        self._krausOperators = listOfKrausOperators
        self._shape = listOfKrausOperators[0].shape

    def shape(self):
        return self._shape


def damping_map(prob=0.1):
    '''
    Amplitude damping.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented
        by the various Kraus operators.

    Returns
    -------
    NoiseModel
        Container of ndarray matrix representations of the Kraus operators in
        the corresponding quantum operation.
    '''

    static = np.array([[1, 0],
                       [0, np.sqrt(1 - prob)]])
    decay = np.array([[0, np.sqrt(prob)],
                      [0, 0]])

    return NoiseModel([static, decay])


def depolarization_map(prob=0.1):
    '''
    Depolarizing channel.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented
        by the various Kraus operators.

    Returns
    -------
    NoiseModel
        Container of ndarray matrix representations of the Kraus operators in
        the corresponding quantum operation.
    '''

    dep_i = np.sqrt(1 - 3.0*prob/4) * np.array([[1, 0],
                                                [0, 1]])
    dep_x = np.sqrt(1.0*prob/4) * np.array([[0, 1],
                                            [1, 0]])
    dep_y = np.sqrt(1.0*prob/4) * np.array([[0, -1j],
                                            [1j, 0]])
    dep_z = np.sqrt(1.0*prob/4) * np.array([[1, 0],
                                            [0, -1]])

    return NoiseModel([dep_i, dep_x, dep_y, dep_z])


def phase_map(prob=0.1):
    '''
    Phase damping.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented
        by the various Kraus operators.

    Returns
    -------
    NoiseModel
        Container of ndarray matrix representations of the Kraus operators in
        the corresponding quantum operation.
    '''

    phase_1 = np.array([[1, 0],
                        [0, np.sqrt(1 - prob)]])
    phase_2 = np.array([[0, 0],
                        [0, np.sqrt(prob)]])

    return NoiseModel([phase_1, phase_2])


def p_flip_map(prob=0.1):
    '''
    Phase flip.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented
        by the various Kraus operators.

    Returns
    -------
    NoiseModel
        Container of ndarray matrix representations of the Kraus operators in
        the corresponding quantum operation.
    '''

    static = np.sqrt(1 - prob) * np.array([[1, 0],
                                           [0, 1]])
    flip = np.sqrt(prob) * np.array([[1, 0],
                                     [0, -1]])

    return NoiseModel([static, flip])


def b_flip_map(prob=0.1):
    '''
    Bit flip.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented
        by the various Kraus operators.

    Returns
    -------
    NoiseModel
        Container of ndarray matrix representations of the Kraus operators in
        the corresponding quantum operation.
    '''

    static = np.sqrt(1 - prob) * np.array([[1, 0],
                                           [0, 1]])
    flip = np.sqrt(prob) * np.array([[0, 1],
                                     [1, 0]])

    return NoiseModel([static, flip])


def bp_flip_map(prob=0.1):
    '''
    Bit-phase flip.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented
        by the various Kraus operators.

    Returns
    -------
    NoiseModel
        Container of ndarray matrix representations of the Kraus operators in
        the corresponding quantum operation.
    '''

    static = np.sqrt(1 - prob) * np.array([[1, 0],
                                           [0, 1]])
    flip = np.sqrt(prob) * np.array([[0, -1j],
                                     [1j, 0]])

    return NoiseModel([static, flip])


def _isListOfIdenticalSizeSquareNumpyArrays(listOfMatrices):
    if not isinstance(listOfMatrices, list):
        raise TypeError("Non-list argument encountered.")

    if not all(_is_numeric_square_matrix(operator)
               and isinstance(operator, type(np.array([])))
               and operator.shape == listOfMatrices[0].shape
               for operator in listOfMatrices):
        return False

    return True


def _isTracePreserving(listOfMatrices):
    matrixDiagonalLength = listOfMatrices[0].shape[0]
    matrixProducts = map(lambda operator:
                         np.matmul(np.conjugate(operator.T), operator),
                         listOfMatrices)
    sumOfMatrixProducts = reduce(
        lambda product1, product2: product1 + product2,
        matrixProducts)

    if not np.allclose(sumOfMatrixProducts, np.eye(matrixDiagonalLength)):
        return False
    else:
        return True


def _validateListOfKrausOperators(listOfKrausOperators):
    if not _isListOfIdenticalSizeSquareNumpyArrays(listOfKrausOperators):
        raise WrongShapeError("List of Kraus operators must be a "
                              "list of numpy "
                              "ndarrays of identical shape.")
    if not _isTracePreserving(listOfKrausOperators):
        raise NormalizationError("List of Kraus operators must be "
                                 "trace-preserving.")
    if len(listOfKrausOperators) < 2:
        raise ValueError("List of Kraus operators must contain "
                         "at least two elements.")
