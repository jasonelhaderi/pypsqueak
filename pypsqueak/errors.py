'''
Custom errors.
'''

class WrongShapeError(ValueError):
    pass

class NullVectorError(ValueError):
    pass

class NormalizationError(ValueError):
    pass

class InhomogenousInputError(TypeError):
    pass

class NonUnitaryInputError(ValueError):
    pass

class UndeclaredGateError(NotImplementedError):
    pass

class UnknownInstruction(NotImplementedError):
    pass
