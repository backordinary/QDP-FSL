# https://github.com/jjhorder/qiskit_scripts/blob/49bd12b2740e44adb00edf18b396bfa64d6c2246/half_adder.py
from qiskit import *
from qiskit.tools.visualization import plot_histogram
import time

# A two-bit half adder of the form a + b, where a is 0 or 1 and b is 0 or 1.

def half_adder(a, b):
    print('Computing %d + %d.\n' % (a, b))

    qc = QuantumCircuit(4, 2)

    # qc[i] = 0, so need to flip according to a, b input
    if a == 1:
        qc.x(0)
    if b == 1:
        qc.x(1)
    qc.barrier()

    # store binary addition on qc[2, 3]
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.ccx(0, 1, 3)
    qc.barrier()

    # read result to classical register
    qc.measure(2, 0)
    qc.measure(3, 1)

    qc.draw(output = 'mpl');
    job = execute(qc, Aer.get_backend('qasm_simulator'))
    hist = job.result().get_counts()
    plot_histogram(hist);


# half_adder(0, 0)
# half_adder(0, 1)
# half_adder(1, 0)
half_adder(1, 1)
