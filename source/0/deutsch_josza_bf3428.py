# https://github.com/kchatr/deutsch-jozsa-algorithm/blob/45bbe99c45418055681572bf2de4ca0eb0ddf990/Deutsch-Josza.py
import qiskit as q
from random import getrandbits

# If the first input is 0, use the oracle for f(x) = 0.
# If the first input is 1, use the oracle for f(x) = 1.
def get_constant_oracle(n, qc):
    if n == 0:
        return
    else:
        qc.x(1)

# If the first input is 0, use the oracle for f(x) = x.
# If the first input is 1, use the oracle for f(x) = !x.
def get_balanced_oracle(n, qc):
    if n == 0:
        qc.cx(0, 1)
    else:
        qc.x(0)
        qc.cx(0, 1)

def get_random_oracle(qc):
    if getrandbits(1) == 0:
        get_constant_oracle(getrandbits(1), qc)
    else:
        get_balanced_oracle(getrandbits(1), qc)

def deutsch_josza():
    # Step 1
    qc = q.QuantumCircuit(2, 2)
    qc.barrier()
    # Step 2
    qc.x(1)
    qc.barrier()
    # Step 3
    qc.h(0)
    qc.h(1)
    qc.barrier()
    # Step 4
    get_random_oracle(qc)
    qc.barrier()
    # Step 5
    qc.h(0)
    qc.h(1)
    qc.barrier()
    # Step 6
    qc.measure(range(2), range(2))
    qc.draw()

