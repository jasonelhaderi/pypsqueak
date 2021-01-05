import numpy as np


def _is_listlike(obj):
    if (type(obj) != list
        and type(obj) != tuple
            and not isinstance(obj, type(np.array([0])))):
        return False
    else:
        return True


def _has_only_numeric_elements(obj):
    '''
    Checks that the elements of ``obj`` are all numeric.
    '''
    for element in obj:
        if hasattr(element, '__iter__'):
            return False
        else:
            try:
                element + 5
            except TypeError:
                return False

    return True


def _is_numeric_square_matrix(some_matrix):
    '''
    Checks that the argument is a numeric, square matrix.
    '''

    try:
        column_length = len(some_matrix)
    except TypeError:
        return False

    if len(some_matrix) == 0:
        return False

    for row in some_matrix:
        if (not _is_listlike(row)
            or not _has_only_numeric_elements(row)
                or (len(row) != column_length)):

            return False

    return True


def _is_unitary(some_matrix):
    '''
    Checks that the argument is a unitary matrix
    '''

    if not _is_numeric_square_matrix(some_matrix):
        return False

    product_with_hermitian_conjugate = np.dot(
        np.array(some_matrix).conj().T,
        some_matrix)

    if not np.allclose(
            product_with_hermitian_conjugate,
            np.eye(len(some_matrix))):
        return False
    else:
        return True


def _is_normalizable(some_vector):

    if all(element == 0 for element in some_vector):
        return False
    else:
        return True


def _is_power_of_two(n):
    '''
    Check whether or not ``n`` is a power of two.
    '''
    if not n == int(n):
        return False

    n = int(n)
    if n == 1:
        return True
    elif n >= 2:
        return _is_power_of_two(n/2.0)
    else:
        return False
