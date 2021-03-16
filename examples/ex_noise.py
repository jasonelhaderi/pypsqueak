import context  # Remove this import if running with pip installed version.

import numpy as np
from pypsqueak.api import qReg, qOp
from pypsqueak.gates import X, I
from pypsqueak.noise import damping_map
import sys

# Prep a qReg in the |1> state
qubit = qReg()
X.on(qubit)

# Send it through an amp decay channel with 0.3 chance of decay.
if len(sys.argv) > 1 and int(sys.argv[1]) > 0:
    n_runs = int(sys.argv[1])
else:
    n_runs = 1000

if len(sys.argv) > 2 and float(sys.argv[2]) >= 0 and float(sys.argv[2]) <= 1:
    prob = float(sys.argv[2])
else:
    prob = 0.3
noisy_channel = qOp(kraus_ops=damping_map(prob))

zeros = 0
ones = 0

print("Sending the state |1> through a noisy channel {} times with amplitude decay probability={}...".format(n_runs, prob))
for i in range(n_runs):
    noisy_channel.on(qubit)
    result = qubit.measure(0)
    if result == 0:
        zeros += 1
        # Reset qReg to |1> for next run.
        X.on(qubit)
    else:
        # No need to reset qReg
        ones += 1

error_bar = 0.5 * np.sqrt(1/n_runs)
print("Observed probabilities")
print("Zero measurement = {0:.3f} +/- {1:.2e}".format(zeros/n_runs, error_bar))
print("One measurement = {0:.3f} +/- {1:.2e}".format(ones/n_runs, error_bar))
