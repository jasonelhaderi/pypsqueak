# pypSQUEAK — Python Packaged Semantic Quantum Expression Architecture
An extension of Python providing high-level object abstractions for quantum devices.

Features of pypSQUEAK include:
* Variable-size quantum registers. The sky's the limit. (Well, your hard drive's size is anyway.)
* Built-in set of universal one-qubit gates as well as several important two-qubit gates.
* User-defined static or parametric gates of arbitrary size.
* Modeling of noisy quantum channels.

For more information, consult the [documentation](https://pypsqueak.readthedocs.io/en/latest/index.html).

## Installation
Installation is done via `pip`:
```pip install pypsqueak```

## Examples
Several examples are provided in the [examples](https://github.com/jasonelhaderi/pypsqueak/tree/master/examples) folder. They are Python scripts demonstrating various aspects of pypSQUEAK.

Here is an example of a script that constructs a pypSQUEAK program to measure a qubit in the |1> state in the presence of noise:
```python
import pypsqueak.api as sq
from pypsqueak.gates import X
from pypsqueak.noise import damping_map

noisy_channel = sq.qOp(kraus_ops=damping_map(0.3))

zeros = 0
ones = 0
n_runs = 100
for i in range(n_runs):
    q = sq.qReg()
    # Prep the |1> state
    X.on(q)
    # Send it through an amp decay channel with 0.3 chance of decay
    noisy_channel.on(q)
    # measure the resulting qubit
    result = q.measure(0)
    if result == 0:
        zeros += 1
    else:
        ones += 1

print(zeros/n_runs, ones/n_runs)
```

## License
This projects is licensed under the MIT License. See LICENSE.txt for more details.
