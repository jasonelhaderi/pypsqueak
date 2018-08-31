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

def is_power_2(n):
    '''
    Helper function for testing validity of Qubit/Gate sizes
    '''
    if not n == int(n):
        return False

    n = int(n)
    if n == 1:
        return True

    elif n >= 2:
        return is_power_2(n/2.0)

    else:
        return False
