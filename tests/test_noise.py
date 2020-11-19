# Standard modules
import unittest
import numpy as np

# pypSQUEAK modules
from pypsqueak.errors import NormalizationError, WrongShapeError
from pypsqueak.noise import NoiseModel


# todo separate out tests on non-noisemodel classes
class NoiseModelValidInput(unittest.TestCase):

    def test_NoiseModelInitializesForValidKrausOpsList(self):
        '''
        ``NoiseModel`` should initialize correctly with a valid list of
        Kraus matrices.
        '''
        damping_map_kraus_ops = [
            np.array([[1, 0],
                      [0, np.sqrt(0.5)]]),
            np.array([[0, np.sqrt(0.5)],
                      [0, 0]])
        ]

        damping_map = NoiseModel(damping_map_kraus_ops)
        self.assertEqual(damping_map.shape(), (2, 2))


class NoiseModelInvalidInput(unittest.TestCase):

    def setUp(self):
        self.test_noise_model = NoiseModel()

    def test_NoiseModelArgumentIsNotAList(self):
        '''
        A ``TypeError`` gets thrown if the argument of
        ``NoiseModel.setKrausOperators`` is not a list.
        '''

        not_lists = ['apple', 3.153]

        for item in not_lists:
            self.assertRaises(
                TypeError,
                self.test_noise_model.setKrausOperators,
                item)

    def test_NoiseModelArgumentListContainsNonArray(self):
        '''
        ``NoiseModel`` throws a ``WrongShapeError`` if any of the elements
        of the ``kraus_ops`` list is not a numpy ndarray.
        '''

        one_not_like_the_other = [
            np.eye(2),
            'nothing to see here'
        ]

        self.assertRaises(WrongShapeError,
                          self.test_noise_model.setKrausOperators,
                          one_not_like_the_other)

    def test_NoiseModelArgElementsShapeMismatch(self):
        '''
        ``NoiseModel`` throws a ``WrongShapeError`` if
        the shape of any of the matricies in the list of Kraus operators don't
        match all the rest.
        '''

        kraus_ops_mismatch = [np.zeros((5, 2)), np.zeros((3, 17))]
        someNoiseModel = NoiseModel()

        self.assertRaises(WrongShapeError,
                          NoiseModel, kraus_ops_mismatch)
        self.assertRaises(WrongShapeError,
                          someNoiseModel.setKrausOperators, kraus_ops_mismatch)

    def test_NoiseModelLessThanTwoKrausOps(self):
        '''
        ``NoiseModel`` throws a ``ValueError`` if the list
        ``kraus_ops`` has less than two elements.
        '''

        single_valid_kraus_op = [
            np.eye(2)
        ]

        self.assertRaises(
            ValueError,
            self.test_noise_model.setKrausOperators,
            single_valid_kraus_op)

    def test_NoiseModelFailsWithNonTracePreservingKrausOps(self):
        '''
        A ``NormalizationError`` is thrown if the provided Kraus
        operators aren't trace-preserving.
        '''

        incomplete_kraus_ops = [
            np.array([[0, 1],
                      [1, 0]]),
            np.array([[1, 0],
                      [1, 0]])
        ]

        self.assertRaises(NormalizationError, NoiseModel, incomplete_kraus_ops)


if __name__ == '__main__':
    unittest.main()
