# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/test_0_local.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer
from qiskit.tools.visualization import circuit_drawer

#Create a Quantum Register with 2 qubits.
q = QuantumRegister(2)
# Create a Classical Register with 2 bits.
c = ClassicalRegister(2)
# Create a Quantum Circuit
qc = QuantumCircuit(q, c)
# Add a H gate on qubit 0, putting this qubit in superposition.
qc.h(q[0])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
qc.cx(q[0], q[1])
# Add a Measure gate to see the state.
qc.measure(q, c)
# Draw the circuit
circuit_drawer(qc, filename="imgs/test.png")
# See a list of available local simulators
print("Aer backends: ", Aer.backends())
# Compile and run the Quantum circuit on a simulator backend
backend_sim = Aer.get_backend('qasm_simulator')
# Execute is just a shortcut for compile and run, i.e.
# qobj = compile(qc, backend_sim, shots=2000)
# job_sim = backend.run(qobj)
# Also note that the call to execute (or run) is non blocking
job_sim = execute(qc, backend_sim)
print("Executing, non blocking")
# Blocking call
result_sim = job_sim.result()
# Show the results
print("simulation: ", result_sim)
print(result_sim.get_counts(qc))
