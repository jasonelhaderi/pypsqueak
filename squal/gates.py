import numpy as np
import cmath

from squal.squalcore import Gate

'''
Functions implementing standard quantum gates on one or more target_qubits, and
classical unary and binary gates. The quantum gates return a tuple consisting of
(Gate object, target_qubits). The classical unary gates take an input_bit and return
an output_bit. The binary gates take input_bit_1 and input_bit_2, and return an
output_bit. COPY and EXCHANGE classical gates are also implemented.
'''

# Quantum gates

def X(target_qubit):

    matrix_rep = [[0, 1],
                  [1, 0]]

    return (Gate(matrix_rep, name='X'), target_qubit)

def Y(target_qubit):

    matrix_rep = [[0, -1j],
                  [1j, 0]]

    return (Gate(matrix_rep, name='Y'), target_qubit)

def Z(target_qubit):

    matrix_rep = [[1, 0],
                  [0, -1]]

    return (Gate(matrix_rep, name='Z'), target_qubit)

def I(target_qubit):

    return (Gate(name='I'), target_qubit)

def H(target_qubit):

    matrix_rep = [[1/np.sqrt(2), ((-1)**i) * 1/np.sqrt(2)] for i in range(2)]

    return (Gate(matrix_rep, name='H'), target_qubit)

def S(target_qubit):

    matrix_rep = [[1, 0],
                  [0, 1j]]

    return (Gate(matrix_rep, name='S'), target_qubit)

def T(target_qubit):

    matrix_rep = [[1, 0],
                  [0, np.exp(1j * np.pi/4)]]

    return (Gate(matrix_rep, name='T'), target_qubit)

def SWAP(target_qubit_i, target_qubit_j):

    matrix_rep = [[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]]

    return (Gate(matrix_rep, name='SWAP'), target_qubit_i, target_qubit_j)

# Classical gates (prepended with '_' are for the backend in QCSim)

def NOT(c_reg_loc):
    return 'NOT', c_reg_loc

def _NOT(input_bit):
    output_bit = 1 - input_bit
    return output_bit

def TRUE(c_reg_loc):
    return 'TRUE', c_reg_loc

def _TRUE(input_bit):
    output_bit = 1
    return output_bit

def FALSE(c_reg_loc):
    return 'FALSE', c_reg_loc

def _FALSE(input_bit):
    output_bit = 0
    return output_bit

def AND(c_reg_loc_1, c_reg_loc_2, save_loc):
    return 'AND', c_reg_loc_1, c_reg_loc_2, save_loc

def _AND(input_bit_1, input_bit_2):
    output_bit = input_bit_1 * input_bit_2
    return output_bit

def OR(c_reg_loc_1, c_reg_loc_2, save_loc):
    return 'OR', c_reg_loc_1, c_reg_loc_2, save_loc

def _OR(input_bit_1, input_bit_2):
    output_bit = 1 - ((1 - input_bit_1) * (1 - input_bit_2))
    return output_bit

def COPY(c_reg_loc_1, c_reg_loc_2):
    # Copies 1 to 2
    return 'COPY', c_reg_loc_1, c_reg_loc_2

def _COPY(input_bit_1, input_bit_2):
    return input_bit_1, input_bit_1

def EXCHANGE(c_reg_loc_1, c_reg_loc_2):
    return 'EXCHANGE', c_reg_loc_1, c_reg_loc_2

def _EXCHANGE(input_bit_1, input_bit_2):
    return input_bit_2, input_bit_1

CLASSICAL_OPS = {'NOT': _NOT,
                 'TRUE': _TRUE,
                 'FALSE': _FALSE,
                 'AND': _AND,
                 'OR': _OR,
                 'COPY': _COPY,
                 'EXCHANGE': _EXCHANGE
                 }

# STD_GATES = {'X': X,
#              'Y': Y,
#              'Z': Z,
#              'I': I,
#              'S': S,
#              'T': T,
#              'SWAP': SWAP}
#
# __all__ = list(STD_GATES.keys())
