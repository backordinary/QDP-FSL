# https://github.com/feisuxiaozhu/gate-complexity/blob/c0246e23c986f7b6f8c6658af7b1d6ca7a56a470/Y120_project/D4/uni-test.py
import numpy as np
from qiskit import *
from qiskit.extensions import *
from qiskit.compiler import transpile

q = QuantumRegister(3, 'q')

U = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]])

qc = QuantumCircuit(q)

# # adjust here to set the initial state
# qc.x(q[0])
# qc.x(q[1])
# qc.x(q[2])

gateU = UnitaryGate(U)
qc.append(gateU, q)

bgates=['u3','cx']
print(qc)
qc_transpiled = transpile(qc, basis_gates=bgates, optimization_level=3)
# qc_transpiled = transpile(qc,  optimization_level=3)
print(qc_transpiled)

# # Create measurement
# meas = QuantumCircuit(3, 3)
# meas.barrier(range(3))
# meas.measure(range(3), range(3))

# # concatenate the mesurement with the circuit
# qc_transpiled_measure = qc + meas



# # Use Aer's qasm_simulator
# backend_sim = Aer.get_backend('qasm_simulator')

# # Execute the circuit on the qasm simulator.
# # We've set the number of repeats of the circuit
# # to be 1024, which is the default.
# job_sim = execute(qc_transpiled_measure, backend_sim, shots=1024)

# # Grab the results from the job.
# result_sim = job_sim.result()
# counts = result_sim.get_counts(qc_transpiled_measure)
# print(counts)


