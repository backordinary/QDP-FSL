# https://github.com/IgnacioRiveraGonzalez/aws_qiskit_notebooks/blob/250b1bed38707af39fd5bb83752c68007c6c552c/.ipynb_checkpoints/grover_qiskit-checkpoint.py
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.providers.aer import AerSimulator
from qiskit import transpile
from qiskit import execute

qubits = 20
oracle = QuantumCircuit(qubits)
oracle.z(0)  # good state = first qubit is |1>
grover_op = GroverOperator(oracle, insert_barriers=True)

sim = AerSimulator(method='statevector')
#circ = transpile(grover_op)
#circ.measure_all()
grover_op.measure_all()
#result = execute(circ, sim, shots=1, blocking_enable=True, blocking_qubits=7).result()

'''def run(qc):
    #backend = Aer.get_backend('aer_simulator')
    sim = AerSimulator(method='statevector')
    job = execute(qc, sim, shots=1, blocking_enable=True, blocking_qubits=2)
    result = job.result()
    return result'''

result = execute(grover_op, sim, shots=1, blocking_enable=True, blocking_qubits=18).result()
#result = run(grover_op)

print(result)
#print(result.get_counts())
print('----------------------------------------------------- \n')