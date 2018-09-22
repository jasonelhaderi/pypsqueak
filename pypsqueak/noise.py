'''
Implements functions returning sets of trace-one Kraus operators. Each function
corresponds to a specific kind of one-qubit noise.

Examples
--------
Here we demonstrate one thousand measurements of the state |1> through a
channel exhibiting amplitude damping with a 30% probability of decay to the |0>
state.

>>> from pypsqueak.api import Program, qcVirtualMachine
>>> from pypsqueak.noise import damping_map
>>> from pypsqueak.gates import I
>>> p = Program()
>>> p.add_instr(X(0), damping_map(0.3))
>>> p.measure(0, 0)
>>> qcvm = qcVirtualMachine()
>>> decays = 0
>>> const = 0
>>> n_trials = 1000
>>> for i in range(n_trials):
...     result = qcvm.execute(p)[0]
...     if result == 1:
...             const += 1
...     else:
...             decays += 1
...
>>> const/n_trials
0.681
>>> decays/n_trials
0.319

'''

import numpy as np

from pypsqueak.squeakcore import Gate
import pypsqueak.gates

def damping_map(prob=0.1):
    '''
    Amplitude damping.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented by
        the various Kraus operators.

    Returns
    -------
    list
        List of ndarray matrix representations of the Kraus operators in the
        corresponding quantum operation.
    '''

    static = np.array([[1, 0],
                       [0, np.sqrt(1 - prob)]])
    decay = np.array([[0, np.sqrt(prob)],
                      [0, 0]])

    return [static, decay]

def depolarization_map(prob=0.1):
    '''
    Depolarizing channel.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented by
        the various Kraus operators.

    Returns
    -------
    list
        List of ndarray matrix representations of the Kraus operators in the
        corresponding quantum operation.
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
    Phase damping.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented by
        the various Kraus operators.

    Returns
    -------
    list
        List of ndarray matrix representations of the Kraus operators in the
        corresponding quantum operation.
    '''

    phase_1 = np.array([[1, 0],
                        [0, np.sqrt(1 - prob)]])
    phase_2 = np.array([[0, 0],
                      [0, np.sqrt(prob)]])

    return [phase_1, phase_2]

def p_flip_map(prob=0.1):
    '''
    Phase flip.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented by
        the various Kraus operators.

    Returns
    -------
    list
        List of ndarray matrix representations of the Kraus operators in the
        corresponding quantum operation.
    '''

    static = np.sqrt(1 - prob) * np.array([[1, 0],
                                       [0, 1]])
    flip = np.sqrt(prob) * np.array([[1, 0],
                                     [0, -1]])

    return [static, flip]

def b_flip_map(prob=0.1):
    '''
    Bit flip.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented by
        the various Kraus operators.

    Returns
    -------
    list
        List of ndarray matrix representations of the Kraus operators in the
        corresponding quantum operation.
    '''

    static = np.sqrt(1 - prob) * np.array([[1, 0],
                                           [0, 1]])
    flip = np.sqrt(prob) * np.array([[0, 1],
                                     [1, 0]])

    return [static, flip]

def bp_flip_map(prob=0.1):
    '''
    Bit-phase flip.

    Parameters
    ----------
    prob : numeric, between 0 and 1
        Probability characterizing the likelihood of the outcomes represented by
        the various Kraus operators.

    Returns
    -------
    list
        List of ndarray matrix representations of the Kraus operators in the
        corresponding quantum operation.
    '''

    static = np.sqrt(1 - prob) * np.array([[1, 0],
                                           [0, 1]])
    flip = np.sqrt(prob) * np.array([[0, -1j],
                                     [1j, 0]])

    return [static, flip]
