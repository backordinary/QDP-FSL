# https://github.com/arshpreetsingh/Qiskit-cert/blob/b2a93d296ee45646bd428570ffa668ea49534398/initilization_vector.py
import qiskit
from qiskit import QuantumCircuit, assemble, Aer

# Create a quantum circuit with 1 qubit and 1 classical bit.
qc = QuantumCircuit(1, 1)

# Initialize the qubit with a state vector of [1, 0] (a value of 0).
state_vector = [1, 0]
qc.initialize(state_vector, 0)

# Apply a NOT gate on qubit 0.
#qc.x(0)

# Measure qubit 0.
qc.measure(0, 0)

job = qiskit.execute(qc, qiskit.BasicAer.get_backend('qasm_simulator'))
#sv_sim = Aer.get_backend('aer_simulator')
print(job.result().get_counts())
#print("State-Vectors below")
#print(job.result().get_statevector())


# Assemble and run the circuit.
sv_sim = Aer.get_backend('qasm_simulator')
qc.save_statevector()
qobj = assemble(qc)
job = sv_sim.run(qobj)

# Get the results of the execution.
ket = job.result().get_statevector()
# Print the results.
print(ket)
