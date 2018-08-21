import pypsqueak.api as sq
from pypsqueak.gates import X, Z, H, CNOT, custom_gate
import numpy as np

# Number of bits in the input bitstring.
n = 2

oracle_type, oracle_value = np.random.randint(2), np.random.randint(2)
oracle_type = 1
# Construct a list of all output values for the oracle.
if oracle_type == 0:
    print("Using a constant oracle with {}-bit input.".format(n))
    value = np.random.randint(2)
    oracle_list = [value for i in range(2**n)]

else:
    print("Using a balanced oracle with {}-bit input.".format(n))
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

# Use the oracle to construct our black box out of the identity matrix.
black_box = np.identity(2**(n+1))

for i in range(len(oracle_list)):
    if oracle_list[i] == 0:
        pass
    else:
        # Swaps |0...> with |1...>
        index_loc = i + 2**n
        if index_loc < 2**(n+1):
            # set original diags to zero
            black_box[i][i] = 0
            black_box[index_loc][index_loc] = 0
            # set off diag elements to one
            black_box[index_loc][i] = 1
            black_box[i][index_loc] = 1
        else:
            pass

# black_box_gate = custom_gate(black_box.tolist(), "ORACLE")

dj_program = sq.Program()

black_box_gate = dj_program.gate_def(black_box.tolist(), "ORACLE")
dj_program.add_instr(black_box_gate())

# Prep qubits 0 to n - 1 in the Hadamard state.
for i in range(n):
    dj_program.add_instr(H(i))

# Flip the answer qubit and apply H.
dj_program.add_instr(X(n))
dj_program.add_instr(H(n))

# Query the oracle.
dj_program.add_instr(black_box_gate())

# Apply H to the first n qubits.
for i in range(n):
    dj_program.add_instr(H(i))

# Measure the first n qubits into the classical register. If the any of the results
# are nonzero, the oracle is balanced. Else, it is constant.
for i in range(n):
    dj_program.measure(i, i)

qc = sq.QCSim()
# print(dj_program)
# Let's try this out a whole bunch of times!
n_tries = 10
zeros = 0
ones = 0
print("Conducting {} trials...".format(n_tries))

for i in range(n_tries):
    result = qc.execute(dj_program)
    print(result)
    if any(result):
        ones += 1
    else:
        zeros += 1
print("Number of zero measurements (constant oracle): {}".format(zeros))
print("Number of one measurements (balanced oracle): {}".format(ones))
