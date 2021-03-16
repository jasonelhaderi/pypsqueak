import context  # Remove this import if running with pip installed version.

from pypsqueak.api import qReg, qOp
from pypsqueak.gates import X, Z, I, ROT_Y, CNOT
from pypsqueak.noise import b_flip_map
import numpy as np
import sys

if len(sys.argv) > 1 and int(sys.argv[1]) > 0:
    n_trials = int(sys.argv[1])
else:
    n_trials = 2000

if len(sys.argv) > 2 and float(sys.argv[2]) <= 1 and float(sys.argv[2]) >= 0:
    prob = float(sys.argv[2])
else:
    prob = 0.1

theory_success = (1 - prob)**3 + 3*prob*(1-prob)**2
theory_failure = 1 - theory_success
successes = 0
failures = 0
noisy_channel = qOp(np.eye(2), kraus_ops=b_flip_map(prob))

# Check that we are getting the correct statistics out of our noisy channel.
print("Initialized noisy channel with {:.1f}% chance of bit flip.".format(100*prob))
print("Probing channel with single qubit {} times...".format(n_trials))
flip_amount = 0
for i in range(n_trials):
    register = qReg()
    noisy_channel.on(register)
    if not np.array_equal([1, 0], register.dump_state()):
        flip_amount += 1

flip_percent = 100*flip_amount/n_trials
print("Bit flip occured ({:.1f} +/- {:.1f})% of the time.\n".format(flip_percent, 0.5*flip_percent/np.sqrt(n_trials)))

print("With bit flip probability of {:.1f}%:".format(100*prob))
print("Theoretical transmission success rate: {:.1f}%".format(100*theory_success))
print("Theoretical transmission failure rate: {:.1f}%\n".format(100*theory_failure))

# Now we send an encoded state through our noisy channel n_trials times.
# Uncomment the print lines in the for loop to peek at the state of the register
# after each operation. Remember, peeking is unphysical!
print("Running {} trials of sending an encoded state...".format(n_trials))
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

print("Successful {:.2f}% of the time".format(100*successes/n_trials))
print("Unsuccessful {:.2f}% of the time".format(100*failures/n_trials))
