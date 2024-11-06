# https://github.com/owainkenwayucl/quantumexperiments/blob/3d3ebb03579775e5df9ba388e0f8e80898b0e611/qiskit_extended_stabilizer/test_sv.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.compiler import assemble
from qiskit.providers.aer import AerError, QasmSimulator
from qiskit.tools.visualization import plot_histogram
import sys
import random   

upper_limit=39
if len(sys.argv) > 1:
   upper_limit = int(sys.argv[1])

print(upper_limit)
circ = QuantumCircuit(upper_limit, upper_limit)

# Initialise with a Hadamard layer
circ.h(range(upper_limit))
# Apply some random CNOT and T gates
qubit_indices = [i for i in range(upper_limit)]
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
control, target, t = random.sample(qubit_indices, 3)
circ.cx(control, target)
circ.t(t)
circ.measure(range(upper_limit), range(upper_limit))

qobj = assemble(circ, backend=QasmSimulator(), shots=1)

es_job = QasmSimulator().run(qobj, backend_options={
        'method': 'statevector'
        })
result = es_job.result()
print('This succeeded?: {}'.format(result.success))
