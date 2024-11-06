# https://github.com/sparverius/simulating-noise-quantum-computing/blob/ad807aa0459c60a21dc6c06ca46937662e1fa2d7/src/load_qasm.py
from qiskit import QuantumCircuit
from qiskit import QiskitError, execute, BasicAer

circ = QuantumCircuit.from_qasm_file("entangled_registers.qasm")
print(circ)

# See the backend
sim_backend = BasicAer.get_backend('qasm_simulator')

# Compile and run the Quantum circuit on a local simulator backend
job_sim = execute(circ, sim_backend)
sim_result = job_sim.result()

# Show the results
print("simulation: ", sim_result)
print(sim_result.get_counts(circ))
