# Standard modules
import unittest
import numpy as np

# pypSQUEAK modules
from pypsqueak.api import qReg, qOp
from pypsqueak.errors import NormalizationError, WrongShapeError
from pypsqueak.noise import damping_map, depolarization_map, b_flip_map


class NoiseInstructionInvalidInput(unittest.TestCase):

    def setUp(self):
        # Test register
        self.test_reg = qReg()
        # Test op
        self.test_op = qOp()

    def test_kraus_ops_not_list(self):
        '''
        A ``TypeError`` gets thrown if the argument of ``qOp.set_noise_model()``
        is not a list.
        '''

        not_lists = ['apple', 3.153, (1, 2, 3)]

        for item in not_lists:
            self.assertRaises(TypeError, self.test_op.set_noise_model, item)

    def test_kraus_ops_not_matricies(self):
        '''
        ``qOp.set_noise_model()`` throws a ``TypeError`` if any of the elements of
        the ``kraus_ops`` list is not a numpy ndarray.
        '''

        one_not_like_the_other = [damping_map(0.5)[0], 'nothing to see here']

        self.assertRaises(TypeError, self.test_op.set_noise_model, one_not_like_the_other)

    def test_kraus_ops_inconsistent_shape(self):
        '''
        The ``qOp.set_noise_model()`` method throws a ``WrongShapeError`` if the
        shape of any of the matricies in in a list of Kraus operators don't
        match all the rest.
        '''

        bad_kraus = [np.zeros((5, 2)), np.zeros((3, 17))]

        self.assertRaises(WrongShapeError, self.test_op.set_noise_model, bad_kraus)

    def test_wrong_number_kraus(self):
        '''
        The ``qOp.set_noise_model()`` method throws a ``TypeError`` if the list
        ``kraus_ops`` has less than two elements.
        '''

        self.assertRaises(TypeError, self.test_op.set_noise_model, ['one'])

    def test_kraus_gate_mismatch(self):
        '''
        An exception gets thrown if the dimensions of the Kraus operators don't
        match the dimensions of the ``qOp`` when calling ``qOp.set_noise_model()``.
        '''

        size_of_two = qOp().kron(qOp())

        self.assertRaises(WrongShapeError, size_of_two.set_noise_model, damping_map(0.5))

    def test_kraus_ops_incomplete(self):
        '''
        A ``NormalizationError`` is thrown if the provided Kraus
        operators aren't trace-preserving.
        '''

        # First construct a list of bad Kraus maps
        bad_kraus_maps = []
        bad_kraus_maps.append([damping_map(0.3)[0], damping_map(0.1)[1]])
        bad_kraus_maps.append([b_flip_map(1.0)[0], depolarization_map(0.5)[0]])

        for bad_kraus in bad_kraus_maps:
            self.assertRaises(NormalizationError,\
                              self.test_op.set_noise_model, kraus_ops=bad_kraus)


if __name__ == '__main__':
    unittest.main()
