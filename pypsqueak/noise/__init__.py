'''
Implements functions returning sets of trace-one Kraus operators. Each function
corresponds to a specific kind of one-qubit noise. For an example of usage, see
:func:`~pypsqueak.api.qOp.set_noise_model`.
'''
from .noise_model import NoiseModel
from .noise_maps import (damping_map, depolarization_map, phase_map,
                         p_flip_map, b_flip_map, bp_flip_map)
