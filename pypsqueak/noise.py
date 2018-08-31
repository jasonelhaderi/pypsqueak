import numpy as np

from pypsqueak.squeakcore import Gate
import pypsqueak.gates
import pypsqueak.api as sq

'''
Implements functions returning sets of trace-one Kraus operators. Each function
corresponds to a specific kind of one-qubit noise.
'''

def damping_map(prob=0.1):
    '''
    Returns Kraus matricies for amplitude damping.
    '''

    static = np.array([[1, 0],
                       [0, np.sqrt(1 - prob)]])
    decay = np.array([[0, np.sqrt(prob)],
                      [0, 0]])

    return [static, decay]

def depolarization_map(prob=0.1):
    '''
    Returns Kraus matricies for a depolarizing channel.
    '''
    dep_i = np.sqrt(1 - 3.0*prob/4) * np.array([[1, 0],
                                                 [0, 1]])
    dep_x = np.sqrt(1.0*prob/4) * np.array([[0, 1],
                                            [1, 0]])
    dep_y = np.sqrt(1.0*prob/4) * np.array([[0, -1j],
                                            [1j, 0]])
    dep_z = np.sqrt(1.0*prob/4) * np.array([[1, 0],
                                            [0, -1]])

    return [dep_i, dep_x, dep_y, dep_z]

def phase_map(prob=0.1):
    '''
    Returns Kraus matricies for phase damping.
    '''

    phase_1 = np.array([[1, 0],
                       [0, np.sqrt(1 - prob)]])
    phase_2 = np.array([[0, 0],
                      [0, np.sqrt(prob)]])

    return [phase_1, phase_2]

def p_flip_map(prob=0.1):
    '''
    Returns Kraus matricies for a phase flip.
    '''

    static = np.sqrt(prob) * np.array([[1, 0],
                                       [0, 1]])
    flip = np.sqrt(1 - prob) * np.array([[1, 0],
                                         [0, -1]])

    return [static, flip]

def b_flip_map(prob=0.1):
    '''
    Returns Kraus matricies for a bit flip.
    '''

    static = np.sqrt(prob) * np.array([[1, 0],
                                       [0, 1]])
    flip = np.sqrt(1 - prob) * np.array([[0, 1],
                                         [1, 0]])

    return [static, flip]

def bp_flip_map(prob=0.1):
    '''
    Returns Kraus matricies for a bit-phase flip.
    '''

    static = np.sqrt(prob) * np.array([[1, 0],
                                       [0, 1]])
    flip = np.sqrt(1 - prob) * np.array([[0, -1j],
                                         [1j, 0]])

    return [static, flip]
