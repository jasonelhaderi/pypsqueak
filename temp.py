import numpy as np
import copy
from pypsqueak.api import qReg, qOp
from pypsqueak.gates import X, Y, Z, I, H, ROT_Y, CNOT
from pypsqueak.errors import WrongShapeError, IllegalCopyAttempt
from pypsqueak.noise import b_flip_map

# Unit test measure_observable
somereg = qReg()
x_val = somereg.measure_observable(X)
print("Result:", x_val)
print("State:", somereg.peek())
##############


a_reg = qReg(1)
X.on(a_reg)
print(a_reg.peek())
X.on(a_reg)
b_reg = qReg(3)
print("Initial state:\t\t", b_reg.peek())
H.on(b_reg, 2)
print("H(2):\t\t\t", b_reg.peek())
H.on(b_reg, 0)
print("H(0):\t\t\t", b_reg.peek())
CNOT.on(b_reg, 1, 0)
print("CNOT(1, 0):\t\t", b_reg.peek())
X.on(b_reg, 2)
double_x = X.kron(I, a_reg)
print(double_x)
b_reg -= 1
print("Remove qubit #2:\t", b_reg.peek())
# Unit test qReg
try:
    copied = copy.copy(a_reg)
except IllegalCopyAttempt:
    print('Copy attempt caught.')
try:
    copied = copy.deepcopy(a_reg)
except IllegalCopyAttempt:
    print('Deepcopy attempt caught.')
a_reg *= b_reg
print(a_reg)
print(a_reg.peek())
print(b_reg)
try:
    print(b_reg.peek())
except NameError:
    print('Register dereferenced successfully.')
##############

# Unit test qOp
my_gate = qOp(np.eye(2))
my_gate.on(a_reg, 1)
try:
    my_gate.on(a_reg, 1, 2)
except WrongShapeError:
    print('Too many targets handled correctly.')

try:
    my_gate.on(b_reg, 3)
except NameError:
    print('Operation on dereferenced register handled correctly.')
##############
print(a_reg.dump_state())

zero = qReg()
one = qReg()
x = qOp([[0, 1],
         [1, 0]])
y = qOp([[0, -1j],
         [1j, 0]])
z = qOp([[1, 0],
         [0, -1]])
i = qOp([[1, 0],
         [0, 1]])
cnot = qOp([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]])
print(x.shape())
# unit test qReg
zero_one = zero * one
##############
print(zero_one, zero_one.peek())
z *= i
print(z)
# unit test qOp
y_squared = y * y
try:
    cnot * i
except WrongShapeError:
    print('Gate size mismatch handled.')
try:
    y_squared *= cnot
except WrongShapeError:
    print('Gate size mismatch handled.')
print(y_squared)
##############

# unit test qReg
try:
    print(zero)
except NameError:
    print('Register zero dereferenced successfully.')

try:
    print(one)
except NameError:
    print('Register one dereferenced successfully.')
##############

# unit test qOp
theta = np.pi/2
print(ROT_Y(theta))
print(ROT_Y(theta).dagger())
print(ROT_Y(theta) * ROT_Y(theta).dagger())
##############
