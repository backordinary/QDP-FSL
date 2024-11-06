# https://github.com/romOlivo/TestQCEC/blob/31ee6623baaf3e48f643879dd60e562cf6cc6f1f/Test/DefaultTest.py
from jkq.qcec import Configuration, Strategy, verify
from qiskit import QuantumCircuit, transpile
from qiskit.test.mock import FakeTokyo

backend = FakeTokyo()

# create your quantum circuit
circ = QuantumCircuit(2)
circ.h(0)
circ.x(0)
circ.h(0)
circ.cx(0, 1)

# append measurements to save output mapping of physical to logical (qu)bits
circ.measure_all()

# compile circuit to appropriate backend using some optimization level
qc_comp = transpile(circ, backend=backend, optimization_level=3)

# set dedicated verification strategy
config = Configuration()
config.strategy = Strategy.compilationflow

# verify the compilation result
result = verify(circ, qc_comp, config)
print(result)
