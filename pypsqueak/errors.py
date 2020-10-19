class WrongShapeError(ValueError):
    '''
    Generally raised when trying to initialize a ``Qubit`` or ``Gate``
    with an improper shape.
    '''
    pass

class NullVectorError(ValueError):
    '''
    Raised when trying to initialize a ``Qubit`` with the null vector.
    '''
    pass

class NormalizationError(ValueError):
    '''
    Raised when the normalization of a ``Qubit`` is broken.
    '''
    pass

class NonUnitaryInputError(ValueError):
    '''
    Raised when trying to instantiate a ``Gate`` with a nonunitary
    argument.
    '''
    pass

class IllegalRegisterReference(NameError):
    '''
    Raised when any kind of operation is attempted on a killed qReg.
    '''
    pass

class IllegalCopyAttempt(NotImplementedError):
    '''
    Raised when a qReg is copied.
    '''
    pass
