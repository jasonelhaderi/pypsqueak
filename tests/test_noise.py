# Standard modules
import unittest
import numpy as np
import cmath

# squal modules
import squal.gates as gt
import squal.api as sq
import squal.errors as sqerr
from squal.noise import damping_map, depolarization_map, b_flip_map

class NoiseInstructionInvalidInput(unittest.TestCase):

    def setUp(self):
        # Test machine
        self.test_qc = sq.QCSim()

        # Test program
        self.test_program = sq.Program()

    def test_kraus_ops_not_list(self):
        '''
        If the kraus_ops argument of add_ninstr() isn't a list, throw a TypeError.
        '''

        not_lists = ['apple', 3.153, gt.I(0)]

        for item in not_lists:
            self.assertRaises(TypeError, self.test_program.add_ninstr, gt.I(0), item)

    def test_kraus_ops_not_matricies(self):
        '''
        add_ninstr() throws a TypeError if any of the elements of the kraus_ops list is not
        a numpy ndarray.
        '''
        one_not_like_the_other = [damping_map(0.5)[0], 'nothing to see here']

        self.assertRaises(TypeError, self.test_program.add_ninstr, gt.I(0),\
                          one_not_like_the_other)

    def test_kraus_ops_inconsistent_shape(self):
        '''
        add_ninstr() throws a WrongShapeError if the shape of any of the matricies in
        kraus_ops doesn't match all the rest.
        '''

        bad_kraus = [np.zeros((5, 2)), np.zeros((3, 17))]

        self.assertRaises(sqerr.WrongShapeError, self.test_program.add_ninstr, gt.I(0),\
                          bad_kraus)

    def test_wrong_number_kraus(self):
        '''
        add_ninstr() throws a ValueError if kraus_ops has less than two elements.
        '''

        self.assertRaises(ValueError, self.test_program.add_ninstr, gt.I(0), ['one'])

    def test_kraus_gate_mismatch(self):
        '''
        add_ninstr() throws a WrongShapeError if the dimensions of the Kraus operators don't
        match the dimensions of the gate.
        '''

        self.assertRaises(sqerr.WrongShapeError, self.test_program.add_ninstr, gt.SWAP(0, 1),\
                          damping_map(0.5))

    def test_kraus_ops_incomplete(self):
        '''
        add_ninstr() should throw a NormalizationError if the provided Kraus
        operators don't satisfy the completeness relation.
        '''

        # First construct a list of bad Kraus maps
        bad_kraus_maps = []
        bad_kraus_maps.append([damping_map(0.3)[0], damping_map(0.1)[1]])
        bad_kraus_maps.append([b_flip_map(1.1)[0], depolarization_map(0.5)[0]])

        for bad_kraus in bad_kraus_maps:
            self.assertRaises(sqerr.NormalizationError,\
                              self.test_program.add_ninstr, gt.I(0), bad_kraus)

if __name__ == '__main__':
    unittest.main()
