import squal.api as sq
from squal.gates import X, I
from squal.noise import damping_map

p = sq.Program()
qc = sq.QCSim()

# Prep the 1 state
p.add_instr(X(2))
# Send it through an amp decay channel with 0.1 chance of decay
p.add_ninstr(I(2), damping_map(0.3))
# measure the resulting qubit
p.measure(2, 0)

zeros = 0
ones = 0
n_runs = 100
for i in range(n_runs):
    if qc.execute(p)[0] == 0:
        zeros += 1
    else:
        ones += 1

print(zeros/n_runs, ones/n_runs)
