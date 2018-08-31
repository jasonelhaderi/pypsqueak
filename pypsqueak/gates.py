import numpy as np
import cmath

from pypsqueak.squeakcore import Gate

'''
Implementations of standard quantum/classical gates on one or more target (qu)bits.
The quantum gate functions return a tuple consisting of the gate's name followed
by any target qubits. Matrix representations of each quantum gate are also provided. For
parametric gates, these representations are functions with a matrix-like return
type. Classical gates have representations as function which return one or more
bit values. The dictionaries STD_GATES and CLASSICAL_OPS map the gate names to the
corresponding representations.
'''

# Pauli Gates

def X(target_qubit):
    return ('X', target_qubit)

_X = [[0, 1],
      [1, 0]]

def Y(target_qubit):
    return ('Y', target_qubit)

_Y = [[0, -1j],
      [1j, 0]]

def Z(target_qubit):
    return ('Z', target_qubit)

_Z = [[1, 0],
      [0, -1]]

def I(target_qubit):
    return ('I', target_qubit)

_I = [[1, 0],
      [0, 1]]

# Hadamard Gate

def H(target_qubit):
    return ('H', target_qubit)

_H = [[1/np.sqrt(2), ((-1)**i) * 1/np.sqrt(2)] for i in range(2)]

# Phase Gates

def PHASE(target_qubit):
    return ('PHASE', target_qubit)

def _PHASE(theta=0):
    matrix_rep = [[1, 0],
                  [0, np.exp(1j * theta)]]
    return matrix_rep

def S(target_qubit):
    return ('S', target_qubit)

_S = [[1, 0],
      [0, 1j]]

def T(target_qubit):
    return ('T', target_qubit)

_T = [[1, 0],
      [0, np.exp(1j * np.pi/4)]]

# Rotation Gates

def RX(target_qubit):
    return ('RX', target_qubit)

def _RX(theta=0):
    matrix_rep = [[np.cos(theta/2.0), -1j*np.sin(theta/2.0)],
                      [-1j*np.sin(theta/2.0), np.cos(theta/2.0)]]
    return matrix_rep

def RY(target_qubit):
    return ('RY', target_qubit)

def _RY(theta=0):
    matrix_rep = [[np.cos(theta/2.0), -np.sin(theta/2.0)],
                  [np.sin(theta/2.0), np.cos(theta/2.0)]]
    return matrix_rep

def RZ(target_qubit):
    return ('RZ', target_qubit)

def _RZ(theta=0):
    matrix_rep = [[np.exp(-1j * theta/2.0), 0],
                  [0, np.exp(1j * theta/2.0)]]
    return matrix_rep

# Two Qubit Gates

def SWAP(target_qubit_i, target_qubit_j):
    return ('SWAP', target_qubit_i, target_qubit_j)

_SWAP = [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]

def CNOT(control_qubit, target_qubit):
    return ('CNOT', control_qubit, target_qubit)

_CNOT = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]]

# Classical gates (prepended with '_' are for the backend in qcVirtualMachine)

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

STD_GATES = {'X': _X,
             'Y': _Y,
             'Z': _Z,
             'I': _I,
             'H': _H,
             'PHASE': _PHASE,
             'S': _S,
             'T': _T,
             'RX': _RX,
             'RY': _RY,
             'RZ': _RZ,
             'SWAP': _SWAP,
             'CNOT': _CNOT
             }
