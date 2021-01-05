from ._helpers import _validateListOfKrausOperators
import numpy as np
from functools import reduce


class NoiseModel:
    '''
    A map characterizing a kind of noise to be simulated in a `qOp`.
    '''

    def __init__(self, kraus_ops=None):
        if kraus_ops is None:
            self._krausOperators = []
            self._shape = None
        else:
            self.setKrausOperators(kraus_ops)

    def getKrausOperators(self):
        return self._krausOperators

    def setKrausOperators(self, listOfKrausOperators):
        _validateListOfKrausOperators(listOfKrausOperators)

        self._krausOperators = listOfKrausOperators
        self._shape = listOfKrausOperators[0].shape

    def shape(self):
        return self._shape

    def __eq__(self, obj):
        if not isinstance(obj, NoiseModel):
            return False
        elif (len(obj._krausOperators) == len(self._krausOperators)
              and reduce(lambda a, b: a and b,
                  [np.array_equal(obj._krausOperators[i], self._krausOperators[i])
                   for i in range(len(self._krausOperators))])):
            return True
        else:
            return False
