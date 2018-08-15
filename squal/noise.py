import numpy as np

from squal.squalcore import Gate
import squal.gates
import squal.api as sq

'''
Quantum operations map a distribution of pure states to another distribution of
pure states. This can be theoretically represented with the evolution of a density
matrix via a set of Kraus operators (also called operation elements). Since the
squal quantum register is implemented as a pure state, we implement quantum
operations by interpreting the coefficients in each term of (sum E*rho*E^{dagger})
as the probability that the operation E gets applied to a state. More precisely,
we take with probability p = <psi|E^{dagger}E|psi> that the transformation
|psi> -> E/sqrt(p)|psi> takes place.
'''

# The following functions generate the list of Kraus matricies for the correponding
# quantum operations.

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

def append_noise(some_gate, kraus_ops):
    # Generates an instruction ('NOISY', kraus_ops, gate_target_tuple).
    # The compiler handles execution of the 'NOISY' instruction by computing
    # the probabilities corresponding to each Kraus operator matrix in the
    # kraus_ops list. Unit test that sum E^{dagger}E <= 1.

    return ('NOISY', kraus_ops, some_gate)
