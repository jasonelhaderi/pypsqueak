'''
Implementations of standard quantum/classical gates acting on one or more target
(qu)bits.

Quantum gates
=============
Functions are provided for creating quantum gate instruction tokens (tuples of
gate name followed by target register locations).

Matrix representations of each quantum gate are also provided for backend use by the
``qcVirtualMachine``. For parametric gates, these representations are functions
with a matrix-like return type that take one or more \*\*kwargs.

Classical gates
===============
Like the quantum gates, the API provides functions for generating instruction
tokens for classical operations. The backend is however slightly different;
each gate is implemented as a function returning one or more classical bit values
when called by the ``qcVirtualMachine``.
'''

import numpy as np
import cmath

from pypsqueak.squeakcore import Gate
from pypsqueak.api import qOp

# Pauli Gates

X = qOp([[0, 1],
         [1, 0]])

# def X(target_qubit):
#     '''
#     Pauli X-gate acting on the qubit in the quantum register indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit to apply the X-gate
#         to.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#
#     return ('X', target_qubit)
#
# _X = [[0, 1],
#       [1, 0]]

Y = qOp([[0, -1j],
         [1j, 0]])

# def Y(target_qubit):
#     '''
#     Pauli Y-gate acting on the qubit in the quantum register indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit to apply the Y-gate
#         to.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#     return ('Y', target_qubit)
#
# _Y = [[0, -1j],
#       [1j, 0]]

Z = qOp([[1, 0],
         [0, -1]])

# def Z(target_qubit):
#     '''
#     Pauli Z-gate acting on the qubit in the quantum register indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit to apply the Z-gate
#         to.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#     return ('Z', target_qubit)
#
# _Z = [[1, 0],
#       [0, -1]]

I = qOp()

# def I(target_qubit):
#     '''
#     One-qubit identity gate acting on the qubit in the quantum register indexed
#     by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit to apply the identity
#         gate to.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#     return ('I', target_qubit)
#
# _I = [[1, 0],
#       [0, 1]]

# Hadamard Gate
H = qOp([[1/np.sqrt(2), ((-1)**i) * 1/np.sqrt(2)] for i in range(2)])
# def H(target_qubit):
#     '''
#     Hadamard gate acting on the qubit in the quantum register indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit to apply the Hadamard
#         gate to.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#     return ('H', target_qubit)
#
# _H = [[1/np.sqrt(2), ((-1)**i) * 1/np.sqrt(2)] for i in range(2)]

# Phase Gates

def PHASE(theta=0):
    return qOp([[1, 0],
                [0, np.exp(1j * theta)]])

# def PHASE(target_qubit):
#     '''
#     Parametric PHASE gate acting on the qubit in the quantum register indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit to apply the Hadamard
#         gate to.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#     return ('PHASE', target_qubit)
#
# def _PHASE(theta=0):
#     matrix_rep = [[1, 0],
#                   [0, np.exp(1j * theta)]]
#     return matrix_rep

S = qOp([[1, 0],
         [0, 1j]])

# def S(target_qubit):
#     '''
#     PHASE gate (theta=pi) acting on the qubit in the quantum register indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit to apply the S gate
#         to.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#     return ('S', target_qubit)
#
# _S = [[1, 0],
#       [0, 1j]]

T = qOp([[1, 0],
         [0, np.exp(1j * np.pi/4)]])

# def T(target_qubit):
#     '''
#     Pi/8 gate acting on the qubit in the quantum register indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit to apply the T gate to.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#     return ('T', target_qubit)
#
# _T = [[1, 0],
#       [0, np.exp(1j * np.pi/4)]]

# Rotation Gates

def ROT_X(theta=0):
    return qOp([[np.cos(theta/2.0), -1j*np.sin(theta/2.0)],
                [-1j*np.sin(theta/2.0), np.cos(theta/2.0)]])

# def RX(target_qubit):
#     '''
#     Parametric X-rotation gate acting on the qubit in the quantum register
#     indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#
#     return ('RX', target_qubit)
#
# def _RX(theta=0):
#     matrix_rep = [[np.cos(theta/2.0), -1j*np.sin(theta/2.0)],
#                       [-1j*np.sin(theta/2.0), np.cos(theta/2.0)]]
#     return matrix_rep

def ROT_Y(theta=0):
    return qOp([[np.cos(theta/2.0), -np.sin(theta/2.0)],
                [np.sin(theta/2.0), np.cos(theta/2.0)]])

# def RY(target_qubit):
#     '''
#     Parametric Y-rotation gate acting on the qubit in the quantum register
#     indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#
#     return ('RY', target_qubit)
#
# def _RY(theta=0):
#     matrix_rep = [[np.cos(theta/2.0), -np.sin(theta/2.0)],
#                   [np.sin(theta/2.0), np.cos(theta/2.0)]]
#     return matrix_rep

def ROT_Z(theta=0):
    return qOp([[np.exp(-1j * theta/2.0), 0],
                [0, np.exp(1j * theta/2.0)]])

# def RZ(target_qubit):
#     '''
#     Parametric Z-rotation gate acting on the qubit in the quantum register
#     indexed by ``target_qubit``.
#
#     Parameters
#     ----------
#     target_qubit : int
#         Nonnegative integer representing the target qubit.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#
#     return ('RZ', target_qubit)
#
# def _RZ(theta=0):
#     matrix_rep = [[np.exp(-1j * theta/2.0), 0],
#                   [0, np.exp(1j * theta/2.0)]]
#     return matrix_rep

# Two Qubit Gates

SWAP = qOp([[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])

# def SWAP(target_qubit_i, target_qubit_j):
#     '''
#     Two-qubit gate swapping qubits in the quantum register. Must be distinct
#     targets.
#
#     Parameters
#     ----------
#     target_qubit_i : int
#         Nonnegative integer representing the first target qubit.
#     target_qubit_j : int
#         Nonnegative integer representing the second target qubit.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#
#     return ('SWAP', target_qubit_i, target_qubit_j)
#
# _SWAP = [[1, 0, 0, 0],
#          [0, 0, 1, 0],
#          [0, 1, 0, 0],
#          [0, 0, 0, 1]]

CNOT = qOp([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]])

# def CNOT(control_qubit, target_qubit):
#     '''
#     Two-qubit controlled NOT gate. Must be distinct targets.
#
#     Parameters
#     ----------
#     control_qubit : int
#         Nonnegative integer representing the control qubit.
#     target_qubit : int
#         Nonnegative integer representing the target qubit.
#
#     Returns
#     -------
#     tuple
#         Consists of the gate name followed by the target qubit location.
#     '''
#
#     return ('CNOT', control_qubit, target_qubit)
#
# _CNOT = [[1, 0, 0, 0],
#          [0, 1, 0, 0],
#          [0, 0, 0, 1],
#          [0, 0, 1, 0]]

# Classical gates (prepended with '_' are for the backend in qcVirtualMachine)

def NOT(c_reg_loc):
    '''
    Classical NOT gate.

    Parameters
    ----------
    c_reg_loc : int
        Nonnegative integer representing which bit in the classical register
        to negate.

    Returns
    -------
    tuple
        Consists of gate name followed by target location.
    '''

    return 'NOT', c_reg_loc

def _NOT(input_bit):
    output_bit = 1 - input_bit
    return output_bit

def TRUE(c_reg_loc):
    '''
    Sets the bit at ``c_reg_loc`` to '1'.

    Parameters
    ----------
    c_reg_loc : int
        Nonnegative integer representing the target bit.

    Returns
    -------
    tuple
        Consists of gate name followed by target location.
    '''

    return 'TRUE', c_reg_loc

def _TRUE(input_bit):
    output_bit = 1
    return output_bit

def FALSE(c_reg_loc):
    '''
    Sets the bit at ``c_reg_loc`` to '0'.

    Parameters
    ----------
    c_reg_loc : int
        Nonnegative integer representing the target bit.

    Returns
    -------
    tuple
        Consists of gate name followed by target location.
    '''

    return 'FALSE', c_reg_loc

def _FALSE(input_bit):
    output_bit = 0
    return output_bit

def AND(c_reg_loc_1, c_reg_loc_2, save_loc):
    '''
    Applies a logical AND gate to two bits in the classical register, and
    saves the result to a third bit.

    Parameters
    ----------
    c_reg_loc_1 : int
        Nonnegative integer representing the first operand bit.
    c_reg_loc_2 : int
        Nonnegative integer representing the second operand bit.
    save_loc : int
        Nonnegative integer representing the save bit

    Returns
    -------
    tuple
        Consists of gate name followed by target locations.
    '''

    return 'AND', c_reg_loc_1, c_reg_loc_2, save_loc

def _AND(input_bit_1, input_bit_2):
    output_bit = input_bit_1 * input_bit_2
    return output_bit

def OR(c_reg_loc_1, c_reg_loc_2, save_loc):
    '''
    Applies a logical OR gate to two bits in the classical register, and
    saves the result to a third bit.

    Parameters
    ----------
    c_reg_loc_1 : int
        Nonnegative integer representing the first operand bit.
    c_reg_loc_2 : int
        Nonnegative integer representing the second operand bit.
    save_loc : int
        Nonnegative integer representing the save bit

    Returns
    -------
    tuple
        Consists of gate name followed by target locations.
    '''

    return 'OR', c_reg_loc_1, c_reg_loc_2, save_loc

def _OR(input_bit_1, input_bit_2):
    output_bit = 1 - ((1 - input_bit_1) * (1 - input_bit_2))
    return output_bit

def COPY(c_reg_loc_1, c_reg_loc_2):
    '''
    Copies a bit in the classical register.

    Parameters
    ----------
    c_reg_loc_1 : int
        Nonnegative integer representing the bit to be copied.
    c_reg_loc_2 : int
        Nonnegative integer representing the location to copy the first bit into.

    Returns
    -------
    tuple
        Consists of instruction name followed by target locations.
    '''

    return 'COPY', c_reg_loc_1, c_reg_loc_2

def _COPY(input_bit_1, input_bit_2):
    return input_bit_1, input_bit_1

def EXCHANGE(c_reg_loc_1, c_reg_loc_2):
    '''
    Swaps two bits in the classical register.

    Parameters
    ----------
    c_reg_loc_1 : int
        Nonnegative integer representing the first bit to be swapped.
    c_reg_loc_2 : int
        Nonnegative integer representing the location of the second bit to be
        swapped.

    Returns
    -------
    tuple
        Consists of instruction name followed by target locations.
    '''
    return 'EXCHANGE', c_reg_loc_1, c_reg_loc_2

def _EXCHANGE(input_bit_1, input_bit_2):
    return input_bit_2, input_bit_1

# CLASSICAL_OPS = {'NOT': _NOT,
#                  'TRUE': _TRUE,
#                  'FALSE': _FALSE,
#                  'AND': _AND,
#                  'OR': _OR,
#                  'COPY': _COPY,
#                  'EXCHANGE': _EXCHANGE
#                  }

# STD_GATES = {'X': _X,
#              'Y': _Y,
#              'Z': _Z,
#              'I': _I,
#              'H': _H,
#              'PHASE': _PHASE,
#              'S': _S,
#              'T': _T,
#              'RX': _RX,
#              'RY': _RY,
#              'RZ': _RZ,
#              'SWAP': _SWAP,
#              'CNOT': _CNOT
#              }
