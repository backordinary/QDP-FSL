# https://github.com/arshpreetsingh/Qiskit-cert/blob/b2a93d296ee45646bd428570ffa668ea49534398/state_vector_draw.py
from qiskit import qiskit, execute, QuantumCircuit, BasicAer
from qiskit.quantum_info import Statevector
from qiskit.tools.monitor import job_monitor
from qiskit.circuit.random import random_circuit
from qiskit.visualization import(
  plot_state_city,
  plot_bloch_multivector,
  plot_state_paulivec,
  plot_state_hinton,
  plot_state_qsphere)
import pylab
# Create a random circuit.
qc = random_circuit(1, 1)
qc.h(0)
#sv = Statevector.from_label('00') # 00 corresponds to the number of qubits
#ev = sv.evolve(qc)
# We can plot using either method!
# plot_state_city(ev)
# ev.draw('city')
# pylab.show()
simulator = BasicAer.get_backend("statevector_simulator")
job = execute(qc, simulator)
job_monitor(job)
state_vector = job.result().get_statevector()
print("Hello!!")
print(state_vector)
plot_bloch_multivector(state_vector)
pylab.show()