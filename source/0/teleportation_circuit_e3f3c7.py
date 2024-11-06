# https://github.com/KobeVK/qiskit--simple-circuits/blob/2c6a769fffc6f5c1496eb3ce7b1593ae4bed654d/qiskit/Teleportation_circuit.py
#A Quantum teleportation ciruit
#Page78 in Robert Loredo's book

# %%
import qiskit
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit import ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit import execute
from matplotlib import style
style.use("dark_background")
# %matplotlib inline

q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q, c)

qc.x(0)
qc.z(0)
qc.barrier()

qc.h(1)
qc.cx(1,2)
qc.barrier()

qc.cx(0,1)
qc.h(0)
qc.measure(0,0)
qc.measure(1,1)
qc.barrier()
qc.cx(1,2)
qc.barrier()

qc.z(2)
qc.x(2)
qc.measure(2,2)

qc.draw(output='mpl')


# backend = Aer.get_backend('qasm_simulator')
# job=execute(qc, backend, shots=1024)
# job_result = job.result()
# results = job_result.get_counts(qc)
# plot_histogram(results)







# %%
