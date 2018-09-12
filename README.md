# pypSQUEAK — Python Packaged Semantic Quantum Expression Architecture
A Python API for writing quantum programs in the SQUEAK language as well as a hybrid quantum/classical virtual machine for running them.

Features of pypSQUEAK include:
* Variable-size quantum and classical registers. The sky's the limit. (Well, your hard drive's size is anyway.)
* Built-in set of universal one-qubit gates as well as several important two-qubit gates.
* User-defined static or parametric gates of arbitrary size.
* Classical logic gates.
* Program control flow with conditionals and loops.
* Modeling of noisy quantum channels.

For more information, consult the [documentation](https://pypsqueak.readthedocs.io/en/latest/index.html).

## Installation
Installation is done via `pip`:
```pip install pypsqueak```

## Examples
Several examples are provided in the [examples](https://github.com/jasonelhaderi/pypsqueak/tree/master/examples) folder. They are Python files demonstrating various aspects of pypSQUEAK. Here is an example of a script that constructs a SQUEAK program to measure a qubit in the one state into the classical register in the presence of noise:
```python
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
```

## License
This projects is licensed under the MIT License. See LICENSE.txt for more details.
