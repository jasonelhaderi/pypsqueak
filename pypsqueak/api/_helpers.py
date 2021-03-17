'''
Internal convenience functions for pypsqueak.api.
'''
import numpy as np


def _makeProjectorOnToSubspace(
        subspaceEigenvalue,
        allEigenvalues,
        matrixOfBasisVectors):
    '''
    Makes a projection operator onto the subspace of Hilbert space
    which corresponds to ``subspaceEigenvalue``, where the columns of
    ``matrixOfBasisVectors`` provide a basis on the Hilbert space (assumed
    to be of dimension ``len(matrixOfBasisVectors)``), and the
    ``i``-th element of the list ``allEigenvalues`` is the eigenvalue
    corresponding to the ``i``-th column of ``matrixOfBasisVectors``.

    Parameters
    ----------
    measurementResult : int or float
        The result of a measurement (must be an element of
        measurementEigenvalues).

    measurementEigenvalues : list
        A list of possible measurement outcome values. The order of this
        list is assumed to match the order of columns in
        measurementTransitionMatrix.

    measurementTransitionMatrix : array
        A square, unitary matrix. The columns of this matrix are assumed
        to be a decomposition of the Hilbert space of the qReg in the
        basis described by all possible measurement outcomes (with respect
        to an observable implicitly defined by the matrix).

    Returns
    -------
    ndarray
        The projection operator onto the measurement
        outcome subspace.
    '''
    dimensionOfHilbertSpace = len(matrixOfBasisVectors)
    projector = np.zeros((
        dimensionOfHilbertSpace,
        dimensionOfHilbertSpace
    ))

    for eigenvalue, state in zip(
            allEigenvalues, matrixOfBasisVectors.T):
        if eigenvalue == subspaceEigenvalue:
            projector += np.outer(state, state)

    return projector


def _flipEndiannessOfBitIterable(iterableOfBits):
    '''
    Returns the iterable `iterableOfBits` reversed in the form of a list,
    casting each element to an integer.
    '''

    return list(map(int, iterableOfBits[::-1]))


def _convertQubitPermToHilbertPerm(
        q_reg,
        qubitPermutationMatrix
):
    '''
    Applies the transformation on computational basis states,
    `qubitPermutationMatrix`, to each `originalBasisLabel` in the Hilbert space
    of the `q_reg`. The result of the transformation, `newBasisLabel`, is used
    in conjunction with the `originalBasisLabel` to determine the Hilbert space
    version of the `qubitPermutationMatrix`.
    '''
    hilbertSpacePermutationMatrix = np.zeros((2**len(q_reg), 2**len(q_reg)))
    for originalBasisLabel in q_reg._qReg__q_reg.computational_decomp():
        newBasisLabelAsListOfInts = np.dot(
            qubitPermutationMatrix,
            _flipEndiannessOfBitIterable(originalBasisLabel))
        newBasisLabel = "".join(
            str(bit) for bit in
            _flipEndiannessOfBitIterable(newBasisLabelAsListOfInts))
        row = int(newBasisLabel, 2)
        col = int(originalBasisLabel, 2)
        hilbertSpacePermutationMatrix[row][col] = 1

    _validatePermutationMatrix(hilbertSpacePermutationMatrix)

    return hilbertSpacePermutationMatrix


def _validatePermutationMatrix(permutationMatrix):
    '''
    Raises an exception if the transpose of `permutationMatrix`
    doesn't invert `permutationMatrix.`
    '''

    if not np.array_equal(
            np.dot(permutationMatrix, permutationMatrix.T),
            np.eye(len(permutationMatrix))):
        raise ValueError("Nonunitary swap encountered.")


def _validateClassicalFunction(func, n, m):
    '''
    Check that the range of `func` is integer, bounded
    below by 0 and above by `2^m - 1`.
    '''
    for value in [func(i) for i in range(2**n)]:
        if not isinstance(value, int):
            raise TypeError(
                'Range of input function contains non-integers.')
        if value < 0 or value > 2**m - 1:
            raise ValueError('Range of input function out of bounds.')
