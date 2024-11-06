# https://github.com/PierreEngelstein/MasterRecherche/blob/8e38d1e8825ec0778059867214c141fc04600860/Programmation/GateBuilding/function_addition.py
from qiskit import *  
from qiskit.circuit.library.standard_gates.z import ZGate
from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates.x import XGate
import numpy as np

qc = QuantumCircuit(4)
# qc.initialize([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3])
# qc += QFT(num_qubits=4, approximation_degree=0, do_swaps=False, inverse=False, insert_barriers=True, name="qft")
# qc += QFT(num_qubits=4, approximation_degree=0, do_swaps=False, inverse=True, insert_barriers=True, name="qft")
qc.append(ZGate().control(2), [1, 2, 3])
qc.append(ZGate().control(2), [0, 2, 3])
qc.append(ZGate().control(1), [2, 3])
qc.append(ZGate().control(3), [0, 1, 2, 3])
qc.append(ZGate().control(2), [0, 2, 3])
qc.append(ZGate().control(3), [0, 1, 2, 3])
qc.append(ZGate().control(2), [0, 1, 3])
# QFT
# qc.h(0)
# qc.h(1)
# qc.h(2)
# qc.h(3)
# qc.x(2)

# qc.barrier()
# qc.h(0)
# qc.cp(np.pi/(2**2), 1, 0)
# qc.cp(np.pi/(2**3), 2, 0)
# qc.cp(np.pi/(2**4), 3, 0)
# qc.barrier()
# qc.h(1)
# qc.cp(np.pi/(2**2), 2, 1)
# qc.cp(np.pi/(2**3), 3, 1)
# qc.barrier()
# qc.h(2)
# qc.cp(np.pi/(2**2), 3, 2)
# qc.h(3)
# qc.barrier()

#CZn gates: sum in the Fourier domain
# qc.cp(np.pi/(2**2), 4, 3)
# qc.cp(np.pi/(2**3), 5, 3)
# qc.cp(np.pi/(2**4), 6, 3)
# qc.barrier()
# qc.cp(np.pi/(2**1), 4, 2)
# qc.cp(np.pi/(2**2), 5, 2)
# qc.cp(np.pi/(2**3), 6, 2)
# qc.barrier()
# qc.cp(np.pi/(2**1), 5, 1)
# qc.cp(np.pi/(2**2), 6, 1)
# qc.barrier()
# qc.cp(np.pi/(2**1), 6, 0)

# Inverse QFT
# qc.h(3)
# qc.barrier()
# qc.cp(np.pi/(2**2), 3, 2)
# qc.h(2)
# qc.barrier()
# qc.cp(np.pi/(2**3), 3, 1)
# qc.cp(np.pi/(2**2), 2, 1)
# qc.h(1)
# qc.barrier()
# qc.cp(np.pi/(2**4), 3, 0)
# qc.cp(np.pi/(2**3), 2, 0)
# qc.cp(np.pi/(2**2), 1, 0)
# qc.h(0)
# qc.barrier()
# qc.measure(3, 0)
# qc.measure(2, 1)
# qc.measure(1, 2)
# qc.measure(0, 3)

qc.draw(output='latex_source', filename='circuit.tex')
print(qc)

result = qiskit.visualization.circuit_drawer(qc, output="text")
print(result)

backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
result = qiskit.execute(qc, backend,shots=shots).result()
counts = result.get_counts(qc)
for res in sorted(counts.int_raw):
    print("{0:04b}".format(res))
print(counts)