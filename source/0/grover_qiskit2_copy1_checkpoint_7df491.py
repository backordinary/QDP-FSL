# https://github.com/IgnacioRiveraGonzalez/aws_qiskit_notebooks/blob/250b1bed38707af39fd5bb83752c68007c6c552c/.ipynb_checkpoints/grover_qiskit2-Copy1-checkpoint.py
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.providers.aer import AerSimulator
from qiskit import transpile
from qiskit import execute
from qiskit.quantum_info import Statevector

qubits = 8
oracle = QuantumCircuit(qubits)
#oracle.cz(0, 1)  # good state = first qubit is |1>
mark_state = Statevector.from_label('10000000')
grover_op = GroverOperator(oracle=mark_state, insert_barriers=True)

sim = AerSimulator(method='statevector')
circ = transpile(grover_op)
#circ.measure_all()
circ.measure_all()
#result = execute(circ, sim, shots=1, blocking_enable=True, blocking_qubits=7).result()

result = execute(circ, sim, shots=1, blocking_enable=True, blocking_qubits=6).result()
#result = run(grover_op)

#print(result)
print(result.get_counts())
print('----------------------------------------------------- \n')