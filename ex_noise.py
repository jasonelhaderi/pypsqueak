import pypsqueak.api as sq
from pypsqueak.gates import X, I
from pypsqueak.noise import damping_map

p = sq.Program()
qcvm = sq.qcVirtualMachine()

# Prep the 1 state
p.add_instr(X(2))
# Send it through an amp decay channel with 0.3 chance of decay
p.add_instr(I(2), damping_map(0.3))
# measure the resulting qubit
p.measure(2, 0)

zeros = 0
ones = 0
n_runs = 100
for i in range(n_runs):
    if qcvm.execute(p)[0] == 0:
        zeros += 1
    else:
        ones += 1

print(zeros/n_runs, ones/n_runs)
