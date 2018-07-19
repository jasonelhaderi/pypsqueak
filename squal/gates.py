import numpy as np
import cmath

from squal.squalcore import Gate

'''
Functions implementing various standard gates on one or more target_qubits.
Each returns a tuple of Gate object, target_qubits.
'''

def X(target_qubit):

    matrix_rep = [[0, 1],
                  [1, 0]]

    return (Gate(matrix_rep), target_qubit)

def Y(target_qubit):

    matrix_rep = [[0, -1j],
                  [1j, 0]]

    return (Gate(matrix_rep), target_qubit)

def Z(target_qubit):

    matrix_rep = [[1, 0],
                  [0, -1]]

    return (Gate(matrix_rep), target_qubit)

def I(target_qubit):

    return (Gate(), target_qubit)

def H(target_qubit):

    matrix_rep = [[1/np.sqrt(2), ((-1)**i) * 1/np.sqrt(2)] for i in range(2)]

    return (Gate(matrix_rep), target_qubit)

def SWAP(target_qubit_i, target_qubit_j):

    matrix_rep = [[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]]

    return (Gate(matrix_rep), target_qubit_i, target_qubit_j)

STD_GATES = {'X': X,
             'Y': Y,
             'Z': Z,
             'I': I,
             'SWAP': SWAP}

__all__ = list(STD_GATES.keys())
