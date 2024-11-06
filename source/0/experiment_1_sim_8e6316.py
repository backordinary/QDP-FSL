# https://github.com/MattiaPeiretti/Quantum-Experiments/blob/21915737df184f5e73663acf750a7f3725a17f6f/experiment_1_SIM.py
# Experiment #1 - Superposition

import time
import numpy as np
import qiskit 
from qiskit.providers.aer import QasmSimulator

CIRCUIT_ITERATIONS = 20

backend = QasmSimulator() # Defining the engine to run the quantum circuit on

Qcircuit = qiskit.QuantumCircuit(1, 1)          # Make a cuircit with 1 QuBit and 1 bit.
Qcircuit.u(np.pi/2, 0, 0, 0)                    # Rotate Q1 to |+⟩.
Qcircuit.measure([0], [0])                      # Measuring the qubit.


# Compiling the circuit
qc_compiled = qiskit.transpile(Qcircuit, backend)

# Runnig the actual circuit
job_sim = backend.run(qc_compiled, shots=CIRCUIT_ITERATIONS)

# Grab the generic results from the job.
result_sim = job_sim.result()

# Sieving for the count results.
counts = result_sim.get_counts(qc_compiled)

print(f"""
Job iterated over {CIRCUIT_ITERATIONS} in {result_sim.time_taken} seconds.
With results
{counts}

""")

# Saving the rapprensetation of the circuit to file.
Qcircuit.draw("mpl", filename='./file.png')

#Generate avg graph of the results.
a = qiskit.visualization.plot_histogram(counts)
a.savefig('avg_results.png')