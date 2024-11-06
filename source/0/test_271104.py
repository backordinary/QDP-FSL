# https://github.com/RupakMukherjee/Poisson-Benchmark/blob/3045c83d9c20342b47e1a332796be0b4b0744d21/test.py
#!pip install -U -r resources/requirements.txt

from IPython.display import clear_output
clear_output()

from qiskit import QuantumCircuit

mycircuit = QuantumCircuit(1)
mycircuit.draw('mpl')

from qiskit.quantum_info import Statevector

sv = Statevector.from_label('0')
sv
sv.data
new_sv = sv.evolve(mycircuit)
new_sv
from qiskit.quantum_info import state_fidelity

state_fidelity(sv, new_sv)

from qiskit.visualization import plot_state_qsphere

plot_state_qsphere(sv.data)

mycircuit = QuantumCircuit(1)
mycircuit.x(0)

mycircuit.draw('mpl')

sv = Statevector.from_label('0')
new_sv = sv.evolve(mycircuit)
new_sv

state_fidelity(new_sv, sv)

plot_state_qsphere(new_sv.data)

sv = Statevector.from_label('0')
mycircuit = QuantumCircuit(1)
mycircuit.h(0)
mycircuit.draw('mpl')

new_sv = sv.evolve(mycircuit)
print(new_sv)
plot_state_qsphere(new_sv.data)

sv = Statevector.from_label('1')
mycircuit = QuantumCircuit(1)
mycircuit.h(0)

new_sv = sv.evolve(mycircuit)
print(new_sv)
plot_state_qsphere(new_sv.data)

# from resources.qiskit_textbook.widgets import gate_demo
# gate_demo(qsphere=True)

sv = Statevector.from_label('00')
plot_state_qsphere(sv.data)

mycircuit = QuantumCircuit(2)
mycircuit.h(0)
mycircuit.cx(0,1)
mycircuit.draw('mpl')

new_sv = sv.evolve(mycircuit)
print(new_sv)
plot_state_qsphere(new_sv.data)

counts = new_sv.sample_counts(shots=1000)

from qiskit.visualization import plot_histogram
plot_histogram(counts)

mycircuit = QuantumCircuit(2, 2)
mycircuit.h(0)
mycircuit.cx(0,1)
mycircuit.measure([0,1], [0,1])
mycircuit.draw('mpl')

from qiskit import Aer, execute
simulator = Aer.get_backend('qasm_simulator')
result = execute(mycircuit, simulator, shots=10000).result()
counts = result.get_counts(mycircuit)
plot_histogram(counts)
