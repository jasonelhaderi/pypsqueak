import squal.api as sq
from squal.gates import X, Z, H
import numpy as np
import math

# balanced oracle is broken. needs to first select target qubit, then
# for each element apply the correct gate. Maybe just construct the
# correct gate by hand?

# Number of bits in the input to the bitstring
n = 2

oracle_type, oracle_value = np.random.randint(2), np.random.randint(2)

oracle_type = 1
if oracle_type == 0:
    print("The oracle returns a constant value.")

else:
    print("The oracle returns a balanced function.")
    oracle_list = []
    n_zeros = (2**n)/2
    n_ones = n_zeros
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
            raise ValueError("Something went wrong with the oracle!")

dj_program = sq.Program()

# Prep qubits 0 to n - 1 in the Hadamard state
for i in range(n):
    dj_program.add_instr(H(i))

# Flip the answer qubit and apply H
dj_program.add_instr(X(n))
dj_program.add_instr(H(n))

# Query the oracle
if oracle_type == 0:    # constant oracle
    if oracle_value == 1:
        # We pick up an overall minus sign (equiv to X answer qubit)
        dj_program.add_instr(X(n))
    else:
        pass

if oracle_type == 1:    # balanced oracle
    for i in range(len(oracle_list)):
        return_value = oracle_list[i]
        if return_value == 0:
            pass
        if return_value == 1:
            # Get quantum register location corresponding to i, then
            # if the target_qubit is zero, apply X(0)Z(0)X(0),
            # or if the target_qubit is not zero, apply Z(target_qubit)
            if i != 0:
                target_qubit = int(math.floor(np.log2(i)))
                dj_program.add_instr(Z(target_qubit))
            else:
                target_qubit = 0
                dj_program.add_instr(X(0))
                dj_program.add_instr(Z(0))
                dj_program.add_instr(X(0))

# Apply H to the first n qubits
for i in range(n):
    dj_program.add_instr(H(i))

# Measure the first n qubits into the classical register. If the result is
# nonzero, the oracle is balanced.
for i in range(n):
    dj_program.measure(i, i)

qc = sq.QCSim()

print(qc.quantum_state(dj_program))
