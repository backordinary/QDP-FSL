# https://github.com/mannurulz/qunatumprogramming/blob/1566602e24c6e428883db9cc3b50b9d44fed366d/qiskit_2qbit_circuit.py
# sample code from IBM Q
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, Aer, IBMQ

## Define a two-qubit quantum circuit
q = QuantumRegister(2, 'q')
c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(q, c)

## Apply the quantum gates
circuit.h(q[0])
circuit.cx(q[0], q[1])

## Finish off with the measurements
circuit.measure(q, c)

## Draw the circuit
circuit.draw(output="mpl")

## First, simulate the circuit

simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, backend=simulator, shots=1024)
result = job.result()

## Then, plot a histogram of the results
counts = result.get_counts(circuit)
from qiskit.tools.visualization import plot_histogram
plot_histogram(counts)

# Next, find the least-busy IBM device
from qiskit.providers.ibmq import least_busy
lb_device = least_busy(IBMQ.backends())

# And run the circuit on that device
job = execute(circuit, backend=lb_device, shots=1024)
from qiskit.tools.monitor import job_monitor
job_monitor(job)
result = job.result()

# Finally, plot a histogram of the results
counts = result.get_counts(circuit)
from qiskit.tools.visualization import plot_histogram
plot_histogram(counts)