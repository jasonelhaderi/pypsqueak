import context

from pypsqueak.api import qReg, qOracle, qOp
from pypsqueak.gates import X, Z, H
import numpy as np

# Number of bits in the input bitstring.
n = 4

oracle_type, oracle_value = np.random.randint(2), np.random.randint(2)
# Construct a list of all output values for the oracle.
if oracle_type == 0:
    print("Using a constant oracle with {}-bit input.".format(n))
    def make_black_box():
        value = np.random.randint(2)
        return [value for i in range(2**n)]

else:
    print("Using a balanced oracle with {}-bit input.".format(n))
    def make_black_box():
        oracle_list = []
        n_zeros = (2**n)/2
        n_ones = n_zeros

        # Append either 0 or 1 randomly subject to the constraint that the total
        # number of zeros and ones be equal.
        for i in range(2**n):
            if n_zeros != 0 and n_ones != 0:
                new_query = np.random.randint(2)
                oracle_list.append(new_query)
                if new_query == 1:
                    n_ones -= 1
                if new_query == 0:
                    n_zeros -= 1

            elif n_zeros == 0 and n_ones != 0:
                oracle_list.append(1)
                n_ones -= 1

            elif n_ones == 0 and n_zeros != 0:
                oracle_list.append(0)
                n_zeros -= 1

            else:
                raise ValueError("The oracle has failed!")

        return oracle_list

# func = lambda i: oracle_list[i]
# # Make the black box.
# black_box = qOracle(func, n)

# Make a function implementing the circuit.
def deutschJozsa(input_register, black_box):
    # Take |input_reg> to |input_reg>|1>
    input_register *= qReg()
    X.on(input_register, 0)

    # Prep qubits 1 to n in the Hadamard state.
    for i in range(1, n+1):
        H.on(input_register, i)

    # Flip the answer qubit and apply H.
    H.on(input_register, 0)

    # Query the oracle.
    black_box.on(input_register, *list(range(n+1)))

    # Apply H to the qubits 1 through n.
    for i in range(1, n+1):
        H.on(input_register, i)

    # Measure the first n qubits. If the any of the results
    # are nonzero, the oracle is balanced. Else, it is constant.
    results = []
    for i in range(1, n+1):
        results.append(input_register.measure(i))

    return results

# Let's try this out a whole bunch of times!
n_tries = 10
zeros = 0
ones = 0

print("Conducting {} trials...".format(n_tries))

for i in range(n_tries):
    input_register = qReg(n)
    oracle_list = make_black_box()
    black_box = qOracle(lambda i: oracle_list[i], n)
    results = deutschJozsa(input_register, black_box)
    if any(results):
        ones += 1
    else:
        zeros += 1
print("Number of zero measurements (constant oracle): {}".format(zeros))
print("Number of one measurements (balanced oracle): {}".format(ones))
