import pytest
import numpy as np

from pypsqueak.errors import NormalizationError, WrongShapeError
from pypsqueak.noise import NoiseModel, b_flip_map


# todo separate out tests on non-noisemodel classes
class TestNoiseModelValidInput:
    @staticmethod
    def compare_noise_models(model_1: NoiseModel, model_2: NoiseModel) -> bool:
        return model_1 == model_2

    def test_NoiseModelInitializesForValidKrausOpsList(self, benchmark):
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

        damping_map = benchmark(NoiseModel, damping_map_kraus_ops)
        assert damping_map.shape() == (2, 2)

    def test_NoiseModelEqualityBitFlip(self, benchmark):
        '''
        Checks that ``NoiseModel`` equality works for equivalent bit flip maps.
        '''
        prob = 0.5
        fair_bit_flip = NoiseModel([
            np.sqrt(prob)*np.array([[1, 0],
                                    [0, 1]]),
            np.sqrt(1 - prob)*np.array([[0, 1],
                                        [1, 0]])
        ])

        modelsAreEqual = benchmark(
            self.compare_noise_models,
            fair_bit_flip,
            b_flip_map(prob)
        )

        assert modelsAreEqual

    def test_NoiseModelInequalityBitFlip(self):
        '''
        Checks that ``NoiseModel`` equality works for inequivalent bit flip
        maps.
        '''
        prob_1 = 0.9
        prob_2 = 0.4
        unfair_bit_flip = [
            np.sqrt(prob_1)*np.array([[1, 0],
                                      [0, 1]]),
            np.sqrt(1 - prob_1)*np.array([[0, 1],
                                          [1, 0]])
        ]

        assert NoiseModel(unfair_bit_flip) != b_flip_map(prob_2)


class TestNoiseModelInvalidInput:
    @pytest.mark.parametrize('non_list', [
        'apple',
        3.153
    ])
    def test_NoiseModelArgumentIsNotAList(self, non_list):
        '''
        A ``TypeError`` gets thrown if the argument of
        ``NoiseModel.setKrausOperators`` is not a list.
        '''
        test_noise_model = NoiseModel()
        with pytest.raises(TypeError):
            test_noise_model.setKrausOperators(non_list)

    def test_NoiseModelArgumentListContainsNonArray(self):
        '''
        ``NoiseModel`` throws a ``WrongShapeError`` if any of the elements
        of the ``kraus_ops`` list is not a numpy ndarray.
        '''
        test_noise_model = NoiseModel()

        one_not_like_the_other = [
            np.eye(2),
            'nothing to see here'
        ]
        with pytest.raises(WrongShapeError):
            test_noise_model.setKrausOperators(one_not_like_the_other)

    def test_NoiseModelArgElementsShapeMismatch(self):
        '''
        ``NoiseModel`` throws a ``WrongShapeError`` if
        the shape of any of the matricies in the list of Kraus operators don't
        match all the rest.
        '''
        kraus_ops_mismatch = [np.zeros((5, 2)), np.zeros((3, 17))]
        someNoiseModel = NoiseModel()

        with pytest.raises(WrongShapeError):
            NoiseModel(kraus_ops_mismatch)
        with pytest.raises(WrongShapeError):
            someNoiseModel.setKrausOperators(kraus_ops_mismatch)

    def test_NoiseModelLessThanTwoKrausOps(self):
        '''
        ``NoiseModel`` throws a ``ValueError`` if the list
        ``kraus_ops`` has less than two elements.
        '''
        test_noise_model = NoiseModel()
        single_valid_kraus_op = [
            np.eye(2)
        ]

        with pytest.raises(ValueError):
            test_noise_model.setKrausOperators(single_valid_kraus_op)

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

        with pytest.raises(NormalizationError):
            NoiseModel(incomplete_kraus_ops)
