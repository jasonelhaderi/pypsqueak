from functools import reduce
import numpy as np
from pypsqueak.errors import NormalizationError, WrongShapeError
from pypsqueak.squeakcore._helpers import _is_numeric_square_matrix


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


def _listOfArraysAreEqual(arr_list_1, arr_list_2):
    areSameLength = len(arr_list_1) == len(arr_list_2)
    listOfElementwiseComparisons = [
        np.array_equal(arr_list_1[i], arr_list_2[i])
        for i in range(len(arr_list_1))
    ]
    elementwiseEquality = reduce(
        lambda a, b: a and b,
        listOfElementwiseComparisons
    )
    if areSameLength and elementwiseEquality:
        return True
    else:
        return False
