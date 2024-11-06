# https://github.com/jrog24/quantum/blob/8a11c07f0cd3393e83ebd5790618433fb941f1f2/circuitbasics.py
# this is the solution from "Circuit Basics" tutorial page from IBM Quantum portal. 
# you can access it directly here: https://qiskit.org/documentation/tutorials/circuits/01_circuit_basics.html

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import QasmSimulator

backend=QasmSimulator()
myCircuit=QuantumCircuit(3)

state=Statevector.from_int(0,2**3)

myCircuit.h(0)
myCircuit.cx(0,1)
myCircuit.cx(0,2)
myCircuit.draw('mpl')


state=state.evolve(myCircuit)
state.draw('latex')
state.draw('qsphere')
meas = QuantumCircuit(3,3)
meas.barrier(range(3))
meas.measure(range(3), range(3))
qc=meas.compose(myCircuit, range(3), front=True)

qc.draw('mpl')
qc_compiled=transpile(qc,backend)
job_sim=backend.run(qc_compiled, shots=1024)

result_sim=job_sim.result()
counts=result_sim.get_counts(qc_compiled)
print(counts)
