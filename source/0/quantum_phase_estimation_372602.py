# https://github.com/slowy07/quantum_computing/blob/ba87291529ef414f6dafb57395c8ecd532ff41f5/quantum_phase_estimation/quantum_phase_estimation.py
import matplotlib.pyplot as plt
import numpy as np
import math

from qiskit import IMBQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from qiskit.visualization import plot_histogram

qpe = QuantumCircuit(4, 3)
qpe.x(3)
qpe.draw()
qpe.h(0)
qpe.h(1)
qpe.h(2)
qpe.draw()

repititions = 1
for i in range(repititions):
    qpe.cp(math.pi / 4, 0, 3)
repititions *= 3
for i in range(repititions):
    qpe.cp(math.pi / 4, 1, 3)
repititions *= 3
for i in range(repititions):
    qpe.cp(math.pi / 4, 2, 3)
repititions *= 3
qpe.draw()

# we apply inverse quantum fourier transformation to convert the state of the
# counting register. here provide the code QFT
def qft_dagger(qc, n):
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-math.pi / float(2 ** (j - m)), m, j)
        qc.h(j)


# well then measure the counting register
qpe.barrier()
# apply inverse QFT
qft_dagger(qpe, 3)
# measure
qpe.barrier()
qpe.measure(0, 0)
qpe.measure(1, 1)
qpe.measure(2, 2)
qpe.draw()

aer_sim = Aer.get_backend("aer_simulator")
shots = 2048
t_qpe = transpile(qpe, aer_sim)
qobj = assemble(t_qpe, shots=shots)
results = aer_sim.run(qobj).results()
answer = results.get_counts()

plot_histogram(answer)

# create and set up circuit
qpe2 = QuantumCircuit(4, 3)
qpe2.h(0)
qpe2.h(1)
qpe2.h(2)

# prepre our eigenstate |psi>
angle = 2 * math.pi / 3
repetitions = 1
for i in range(repetitions):
    qpe2.cp(angle, 0, 3)
repetitions *= 2
for i in range(repetitions):
    qpe2.cp(angle, 1, 3)
repetitions *= 2
for i in range(repetitions):
    qpe2.cp(angle, 2, 3)
repetitions *= 2

# do the inverse QFT
qft_dagger(qpe2, 3)
qpe2.measure(0, 0)
qpe2.measure(1, 1)
qpe2.measure(2, 2)
qpe2.draw()

# let's see the result
aer_sim = Aer.get_backend("aer_simulator")
shots = 4096
t_qpe2 = transpile(qpe2, aer_sim)
qobj = assemble(t_qpe2, shots=shots)
results = aer_sim.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)

# create and set up circuit
qpe3 = QuantumCircuit(6, 5)
qpe3.h(0)
qpe3.h(1)
qpe3.h(2)
qpe3.h(3)
qpe3.h(4)

# prepare our eigenstate |psi>
qpe3.x(5)

# do the controlled u operations
angle = 2 * math.pi / 3
repetitions = 1
for i in range(repetitions):
    qpe3.cp(angle, 0, 5)
repetitions *= 2
for i in range(repetitions):
    qpe3.cp(angle, 1, 5)
repetitions *= 2
for i in range(repetitions):
    qpe3.cp(angle, 2, 5)
repetitions *= 2
for i in range(repetitions):
    qpe3.cp(angle, 3, 5)
repetitions *= 2
for i in range(repetitions):
    qpe3.cp(angle, 4, 5)
repetitions *= 2

# do the inverse QFT
qft_dagger(qpe3, 5)

# measure of course
qpe3.barrier()
qpe3.measure(0, 0)
qpe3.measure(1, 1)
qpe3.measure(2, 2)
qpe3.measure(3, 3)
qpe3.measure(4, 4)

qpe3.draw()

# let's see the result
aer_sim = Aer.get_backend("aer_simulator")
shots = 4096
t_qpe3 = transpile(qpe3, aer_sim)
qobj = assemble(t_qpe3, shots=shots)
results = aer_sim.run(qobj).result()
answer = results.get_counts()

plot_histogram(answer)

qpe.draw()
