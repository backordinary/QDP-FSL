# https://github.com/epiqc/PartialCompilation/blob/50d80f56efdf754e40a0b1dd00404788a03fdf3d/qiskit-terra/examples/python/load_qasm.py
"""
Example on how to load a file into a QuantumCircuit

"""
from qiskit import QuantumCircuit
from qiskit import QiskitError, execute, BasicAer

try:
    circ = QuantumCircuit.from_qasm_file("examples/qasm/entangled_registers.qasm")
    print(circ.draw())

    # See the backend
    sim_backend = BasicAer.get_backend('qasm_simulator')


    # Compile and run the Quantum circuit on a local simulator backend
    job_sim = execute(circ, sim_backend)
    sim_result = job_sim.result()

    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts(circ))

except QiskitError as ex:
    print('There was an internal Qiskit error. Error = {}'.format(ex))

