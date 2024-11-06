# https://github.com/rohitganti/quanutm_computing-/blob/235dd44ad64de21194faac184b584d19559b66c3/quantum.py
import numpy as np
from qiskit import *
#%matplotlib inline


circ = QuantumCircuit(3)

circ.h(0)
circ.cx(0,1)
circ.cx(0,2)

circ.draw('mpl')

from qiskit import Aer
backend= Aer.get_backend('statevector_simulator')

job= execute(circ,backend)
result= job.result()
outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)

from qiskit.visualization import plot_state_city
plot_state_city(outputstate)

backend = Aer.get_backend('unitary_simulator')
job = execute(circ, backend)
result = job.result()
print(result.get_unitary(circ,decimals=3))

meas= QuantumCircuit(3,3)
meas.barrier(range(3))
meas.measure(range(3), range(3))
qc= circ+meas
qc.draw()

backend_sim= Aer.get_backend('qasm_simulator')
job_sim= execute(qc,backend_sim,shots=1024)
result_sim= job_sim.result()

counts= result_sim.get_counts(qc)
print(counts)

from qiskit.visualization import plot_histogram
plot_histogram(counts)

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor


bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)
meas = QuantumCircuit(2, 2)
meas.measure([0,1], [0,1])
backend = BasicAer.get_backend('qasm_simulator') 
circ = bell + meas
result = execute(circ, backend, shots=1000).result()
counts  = result.get_counts(circ)
print(counts)

plot_histogram(counts)

second_result= execute(circ,backend,shots=1000).result()
second_counts= second_result.get_counts(circ)
legend=['First execution','Second execution']
plot_histogram([counts,second_counts],legend=legend)

plot_histogram([counts, second_counts], legend=legend, sort='desc', figsize=(15,12),color=['orange', 'black'], bar_labels=False)

from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere

backend = BasicAer.get_backend('statevector_simulator') 
result = execute(bell, backend).result()
psi  = result.get_statevector(bell)

plot_state_city(psi)

plot_state_hinton(psi)

plot_state_qsphere(psi)

plot_state_paulivec(psi)

plot_bloch_multivector(psi)

plot_state_city(psi, title="My City", color=['black', 'orange'])


plot_state_hinton(psi, title="My Hinton")
