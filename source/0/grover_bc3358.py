# https://github.com/whnbaek/QC2020/blob/256bd6889de01f117fe8aa7730d1bfcc0414f8bd/hw6/grover.py
from qiskit import(
    QuantumCircuit,
    execute,
    Aer)
from qiskit.visualization import plot_histogram
from qiskit.extensions import UnitaryGate
import numpy as np

Uf = np.zeros((16, 16))
target = 0b111
for i in range(16):
	if (i >> 1) == target:
		Uf[i ^ 0b1, i] = 1
	else:
		Uf[i, i] = 1
A = np.full((8, 8), 1 / 8)
I = np.identity(8)

# 1 time
circuit1 = QuantumCircuit(4, 3)
circuit1.h(1)
circuit1.h(2)
circuit1.h(3)
circuit1.barrier()

circuit1.x(0)
circuit1.h(0)
circuit1.unitary(UnitaryGate(Uf), range(4), 'Uf')
circuit1.unitary(UnitaryGate(2 * A - I), range(1, 4), '2A-I')
circuit1.barrier()

circuit1.measure(range(1, 4), range(3))

simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit1, simulator, shots = 1000)

result = job.result()
counts = result.get_counts(circuit1)
print("Total counts (1 time): ", counts)

draw = circuit1.draw(output = 'mpl')
draw.savefig('grover_circuit1.png', bbox_inches = 'tight')
hist = plot_histogram(counts)
hist.savefig('grover_histogram1.png', bbox_inches = 'tight')

# 2 time
circuit2 = QuantumCircuit(5, 3)
circuit2.h(2)
circuit2.h(3)
circuit2.h(4)
circuit2.barrier()

circuit2.x(0)
circuit2.h(0)
circuit2.unitary(UnitaryGate(Uf), [0, 2, 3, 4], 'Uf')
circuit2.unitary(UnitaryGate(2 * A - I), range(2, 5), '2A-I')
circuit2.barrier()

circuit2.x(1)
circuit2.h(1)
circuit2.unitary(UnitaryGate(Uf), range(1, 5), 'Uf')
circuit2.unitary(UnitaryGate(2 * A - I), range(2, 5), '2A-I')
circuit2.barrier()

circuit2.measure(range(2, 5), range(3))

simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit2, simulator, shots = 1000)

result = job.result()
counts = result.get_counts(circuit2)
print("Total counts (2 times): ", counts)

draw = circuit2.draw(output = 'mpl')
draw.savefig('grover_circuit2.png', bbox_inches = 'tight')
hist = plot_histogram(counts)
hist.savefig('grover_histogram2.png', bbox_inches = 'tight')
