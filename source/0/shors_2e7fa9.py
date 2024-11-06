# https://github.com/IshfaqKhawaja/Quantum-Computing/blob/804bdf21a0dea7703929db04127d7a2ef141d51d/shors.py
from qiskit.aqua.algorithms import Shor
from qiskit.aqua import QuantumInstance
from qiskit import Aer
key = 21
base = 2
backend = Aer.get_backend('qasm_simulator')
qi = QuantumInstance(backend=backend, shots=1024)
shors = Shor(N=key, a=base, quantum_instance=qi)
results = shors.run()
print(results)
