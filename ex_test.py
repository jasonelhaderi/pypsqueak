import pypsqueak.api as sq
import pypsqueak.gates
import numpy as np

p = sq.Program()

def my_gate(theta):
    rep = [[np.cos(theta/2), -np.sin(theta/2)],
           [np.sin(theta/2), np.cos(theta/2)]]

    return rep

rotation = p.gate_def(my_gate, "CLS_ROT")
params = [np.pi]
p.add_instr(rotation(params, 0))
p.measure(0, 0)
print(p)

qc = sq.QCSim()
print('Classical register: ', qc.execute(p))
print('Quantum register: ', qc.quantum_state(p))
