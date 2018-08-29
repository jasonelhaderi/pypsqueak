# Standard modules
import unittest
import numpy as np
import cmath

# SQUEAK modules
from pypsqueak.squeakcore import Gate
import pypsqueak.gates as gt
import pypsqueak.api as sq
import pypsqueak.errors as sqerr
from pypsqueak.noise import damping_map, depolarization_map, b_flip_map

class NoiseInstructionInvalidInput(unittest.TestCase):

    def setUp(self):
        # Test program
        self.test_program = sq.Program()

        # Test machine
        self.test_qcvm = sq.qcVirtualMachine()

    def test_kraus_ops_not_list(self):
        '''
        If the kraus_ops argument of _qcVirtualMachine__instr() isn't a list, throw a TypeError.
        '''

        I = Gate(gt._I)

        not_lists = ['apple', 3.153, I]

        for item in not_lists:
            self.assertRaises(TypeError, self.test_qcvm._qcVirtualMachine__instr,\
                              I, 0, kraus_ops=item)

    def test_kraus_ops_not_matricies(self):
        '''
        _qcVirtualMachine__instr()() throws a TypeError if any of the elements of
        the kraus_ops list is not a numpy ndarray.
        '''
        one_not_like_the_other = [damping_map(0.5)[0], 'nothing to see here']

        self.assertRaises(TypeError, self.test_qcvm._qcVirtualMachine__instr, Gate(gt._I), 0,\
                          one_not_like_the_other)

    def test_kraus_ops_inconsistent_shape(self):
        '''
        _qcVirtualMachine__instr() throws a WrongShapeError if the shape of any of the matricies in
        kraus_ops doesn't match all the rest.
        '''

        bad_kraus = [np.zeros((5, 2)), np.zeros((3, 17))]
        I = Gate(gt._I)

        self.assertRaises(sqerr.WrongShapeError, self.test_qcvm._qcVirtualMachine__instr, I, 0,\
                          kraus_ops=bad_kraus)

    def test_wrong_number_kraus(self):
        '''
        _qcVirtualMachine__instr() throws a ValueError if kraus_ops has less than two elements.
        '''

        self.assertRaises(ValueError, self.test_qcvm._qcVirtualMachine__instr,\
                          Gate(gt._I), 0, kraus_ops=['one'])

    def test_kraus_gate_mismatch(self):
        '''
        _qcVirtualMachine__instr() throws a WrongShapeError if the dimensions of the Kraus operators don't
        match the dimensions of the gate.
        '''
        SWAP = Gate(gt._SWAP)

        self.assertRaises(sqerr.WrongShapeError, self.test_qcvm._qcVirtualMachine__instr, SWAP,\
                          0, 1, kraus_ops=damping_map(0.5))

    def test_kraus_ops_incomplete(self):
        '''
        _qcVirtualMachine__instr() should throw a NormalizationError if the provided Kraus
        operators don't satisfy the completeness relation.
        '''

        # First construct a list of bad Kraus maps
        bad_kraus_maps = []
        bad_kraus_maps.append([damping_map(0.3)[0], damping_map(0.1)[1]])
        bad_kraus_maps.append([b_flip_map(1.0)[0], depolarization_map(0.5)[0]])
        I = Gate(gt._I)

        for bad_kraus in bad_kraus_maps:
            self.assertRaises(sqerr.NormalizationError,\
                              self.test_qcvm._qcVirtualMachine__instr, I, 0, kraus_ops=bad_kraus)

if __name__ == '__main__':
    unittest.main()
