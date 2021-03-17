'''
Static gates
============
Pre-defined ``qOp`` objects are provided implementing the common static gates
``X``, ``Y``, ``Z``, ``I``, ``H``, ``S`` (phase), and ``T`` (pi/8).

Parametric gates
================
Common parametric gates (such as rotation operators) are implemented here as
functions returning a ``qOp`` object.
'''

import numpy as np

from pypsqueak.api import qOp

# Pauli Gates
X = qOp([[0, 1],
         [1, 0]])

Y = qOp([[0, -1j],
         [1j, 0]])

Z = qOp([[1, 0],
         [0, -1]])

I = qOp()

# Hadamard gate
H = qOp([[1/np.sqrt(2), ((-1)**i) * 1/np.sqrt(2)] for i in range(2)])


# PHASE gate
def PHASE(theta=0):
    return qOp([[1, 0],
                [0, np.exp(1j * theta)]])


S = qOp([[1, 0],
         [0, 1j]])

T = qOp([[1, 0],
         [0, np.exp(1j * np.pi/4)]])


# Rotation gates
def ROT_X(theta=0):
    return qOp([[np.cos(theta/2.0), -1j*np.sin(theta/2.0)],
                [-1j*np.sin(theta/2.0), np.cos(theta/2.0)]])


def ROT_Y(theta=0):
    return qOp([[np.cos(theta/2.0), -np.sin(theta/2.0)],
                [np.sin(theta/2.0), np.cos(theta/2.0)]])


def ROT_Z(theta=0):
    return qOp([[np.exp(-1j * theta/2.0), 0],
                [0, np.exp(1j * theta/2.0)]])


# Two-qubit gates
SWAP = qOp([[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]])

CNOT = qOp([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]])
