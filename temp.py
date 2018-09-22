import numpy as np
import copy
from pypsqueak.api import qReg, qOp
from pypsqueak.gates import X, Y, Z, I, H, ROT_Y, CNOT
from pypsqueak.errors import WrongShapeError, IllegalCopyAttempt
from pypsqueak.noise import b_flip_map

### BIT FLIP CODE DEMO ###
prob = 0.1
n_trials = 1000
theory_success = (1 - prob)**3 + 3*prob*(1-prob)**2
theory_failure = 1 - theory_success
print("With bit flip probability {}...".format(prob))
print("Theoretical success rate: {:.1f}%\n".format(100*theory_success))
print("Theoretical failure rate: {:.1f}%\n".format(100*theory_failure))
successes = 0
failures = 0
noisy_channel = qOp(np.eye(2), kraus_ops=b_flip_map(prob))

for i in range(n_trials):
    # Initialize a state.
    super_position = qReg(1)
    ROT_Y(0.2).on(super_position)
    # print("Input state |psi> =", super_position.peek())
    # Encode against bit flip.
    CNOT.on(super_position, 1, 0)
    CNOT.on(super_position, 2, 0)
    init_state = super_position.dump_state()
    # print("Encoded state |psi'> =", super_position.peek())
    # Send state through noisy channel.
    for qubit in range(len(super_position)):
        noisy_channel.on(super_position, qubit)
    # print("Encoded state after noisy transmission:", super_position.peek())
    # Diagnose error syndrome.
    Z_21 = Z.kron(Z, I)
    Z_10 = I.kron(Z, Z)
    product_21 = super_position.measure_observable(Z_21)
    # print("Action of Z_21:", super_position.peek())
    # print("Z_21 measurement:", product_21)
    product_10 = super_position.measure_observable(Z_10)
    # print("Action of Z_10:", super_position.peek())
    # print("Z_10 measurement:", product_10)
    if product_10 == product_21:
        if product_10 == 1:
            # No correction required (1 - p)^3
            pass
        else:
            # Middle qubit flipped (1 - p)^2 * p
            X.on(super_position, 1)
    if product_10 != product_21:
        if product_21 == -1:
            # Qubit 2 flipped (1 - p)^2 * p
            X.on(super_position, 2)
        else:
            # Qubit 0 flipped (1 - p)^2 * p
            X.on(super_position, 0)
    # print("Recovered state:", super_position.peek())
    if np.allclose(init_state, super_position.dump_state()):
        successes += 1
    else:
        failures += 1

print("With {} trials...".format(n_trials))
print("Successful {:.1f}% of the time".format(100*successes/n_trials))
print("Unsuccessful {:.1f}% of the time".format(100*failures/n_trials))

flip_amount = 0
for i in range(n_trials):
    register = qReg()
    noisy_channel.on(register)
    if not np.array_equal([1, 0], register.dump_state()):
        flip_amount += 1

flip_percent = 100*flip_amount/n_trials
print("Bit flip occuring {:.1f} +/- {:.1f}% of the time.".format(flip_percent, flip_percent/np.sqrt(n_trials)))
### BIT FLIP CODE DEMO END ###

# somereg = qReg()
# x_val = somereg.measure_observable(X)
# print("Result:", x_val)
# print("State:", somereg.peek())



# a_reg = qReg(1)
# X.on(a_reg)
# print(a_reg.peek())
# X.on(a_reg)
# b_reg = qReg(3)
# print("Initial state:\t\t", b_reg.peek())
# H.on(b_reg, 2)
# print("H(2):\t\t\t", b_reg.peek())
# H.on(b_reg, 0)
# print("H(0):\t\t\t", b_reg.peek())
# CNOT.on(b_reg, 1, 0)
# print("CNOT(1, 0):\t\t", b_reg.peek())
# X.on(b_reg, 2)
# double_x = X.kron(I, a_reg)
# print(double_x)
# b_reg -= 1
# print("Remove qubit #2:\t", b_reg.peek())
# try:
#     copied = copy.copy(a_reg)
# except IllegalCopyAttempt:
#     print('Copy attempt caught.')
# try:
#     copied = copy.deepcopy(a_reg)
# except IllegalCopyAttempt:
#     print('Deepcopy attempt caught.')
# a_reg *= b_reg
# print(a_reg)
# print(a_reg.peek())
# print(b_reg)
# try:
#     print(b_reg.peek())
# except NameError:
#     print('Register dereferenced successfully.')
#
# my_gate = qOp(np.eye(2))
# my_gate.on(a_reg, 1)
# try:
#     my_gate.on(a_reg, 1, 2)
# except WrongShapeError:
#     print('Too many targets handled correctly.')
#
# try:
#     my_gate.on(b_reg, 3)
# except NameError:
#     print('Operation on dereferenced register handled correctly.')
#
# print(a_reg.dump_state())
#
# zero = qReg()
# one = qReg()
# x = qOp([[0, 1],
#          [1, 0]])
# y = qOp([[0, -1j],
#          [1j, 0]])
# z = qOp([[1, 0],
#          [0, -1]])
# i = qOp([[1, 0],
#          [0, 1]])
# cnot = qOp([[1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 0, 1],
#             [0, 0, 1, 0]])
# print(x.shape())
# zero_one = zero * one
# print(zero_one, zero_one.peek())
# z *= i
# print(z)
# y_squared = y * y
# try:
#     cnot * i
# except WrongShapeError:
#     print('Gate size mismatch handled.')
# try:
#     y_squared *= cnot
# except WrongShapeError:
#     print('Gate size mismatch handled.')
# print(y_squared)
# try:
#     print(zero)
# except NameError:
#     print('Register zero dereferenced successfully.')
#
# try:
#     print(one)
# except NameError:
#     print('Register one dereferenced successfully.')
#
# theta = np.pi/2
# print(ROT_Y(theta))
# print(ROT_Y(theta).dagger())
# print(ROT_Y(theta) * ROT_Y(theta).dagger())
