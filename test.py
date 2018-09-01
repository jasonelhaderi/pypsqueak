import pypsqueak.api as sq
import pypsqueak.gates as gt

q = sq.qcVirtualMachine()
p = sq.Program()

p.add_instr(gt.X(0))
p.measure(0, 0)
loop = sq.Program()
loop.add_instr(gt.X(0))
loop.measure(0, 0)

p.while_loop(0, loop)
then_branch = sq.Program()
then_branch.add_instr(gt.X(1))
p.if_then_else(0, then_branch)
p.add_instr(gt.SWAP(0, 1))
p.measure(1, 1)

print(p)
print(q.quantum_state(p))
print(q.execute(p))
