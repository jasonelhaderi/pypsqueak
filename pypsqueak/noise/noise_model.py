from ._helpers import (_validateListOfKrausOperators,
                       _listOfArraysAreEqual)


class NoiseModel:
    '''
    A map characterizing a kind of noise to be simulated in a `qOp`.
    Takes a list of numpy ndarrays as creation argument, or argument
    to ``NoiseModel.setKrausOperators()``. The elemnts of the list
    must all have the same dimension, and collectively be
    trace-preserving.
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
        elif _listOfArraysAreEqual(obj._krausOperators, self._krausOperators):
            return True
        else:
            return False
