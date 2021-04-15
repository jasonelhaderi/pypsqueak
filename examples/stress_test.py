import context # Remove this import if running with pip installed version.
from functools import reduce
import time

from pypsqueak.api import qReg, qOp
from pypsqueak.gates import I, H, X, Y, Z


def printTimingInfo(start, stop, num_ops=None):
    total_time_microsec = (stop - start) * 10 ** 6
    print("Total time: {:.2e}μs".format(total_time_microsec))
    if num_ops is not None:
        print("Avg time: {:.2e}μs".format(total_time_microsec/num_ops))

# Stress tests to implement:


# kron to make a large gate (take convenience features into consideration)
def kronChain():
    '''
    Checks timing on qOp.kron() when constructing a qOp with a chain of krons.
    '''
    print("Testing a chain of kronecker products")
    start = time.time()
    H.kron(H).kron(H).kron(H).kron(H).kron(H)
    stop = time.time()
    printTimingInfo(start, stop, num_ops=5)


def kronMultipleArgs():
    '''
    Checks timing on qOp.kron() when constructing a qOp with a chain of krons.
    '''
    print("Testing a kronecker product with multiple args")
    start = time.time()
    H.kron(H, H, H, H, H)
    stop = time.time()
    printTimingInfo(start, stop, num_ops=5)


# multiplication of multiple qOps together
def qOpMultiplicationChain(size):
    '''
    Checks timing on multiplication of 6 qOps of some size.
    '''
    print("Multiplying 6 qOps of size {} together".format(size))
    full_gate = reduce(
        lambda a, b: a.kron(b),
        [H for i in range(size)]
    )
    start = time.time()
    full_gate * full_gate * full_gate * full_gate * full_gate
    stop = time.time()
    printTimingInfo(start, stop, num_ops=5)

# application of qOps of various sizes to qReg
# measurement of qReg
# ops between qRegs


if __name__ == '__main__':
    kronChain()
    kronMultipleArgs()
    [qOpMultiplicationChain(i ** 2) for i in range(1, 4)]
