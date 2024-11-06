# https://github.com/happybored/pennylane_vqe/blob/0fd9119ad1283b0add110738ff0bb789032624ca/noise_circuit_qiskit.py
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
import pennylane as qml
import torch


dev = qml.device('default.qubit', wires=2)

theta = Parameter('θ')

qc = QuantumCircuit(2)
qc.rz(theta, [0])
qc.rx(theta, [0])
qc.cx(0, 1)

@qml.qnode(dev,interface='torch')
def circuit(x):
    qml.from_qiskit(qc)({theta: x})
    return qml.expval(qml.PauliZ(0))

angle = torch.tensor(0.011, requires_grad=True)
steps = 10

def cost(x):
    return circuit(x)

opt = torch.optim.Adam([angle], lr = 0.1)

def closure():
    opt.zero_grad()
    loss = cost(angle)
    loss.backward()
    return loss

for i in range(steps):
    opt.step(closure)
    print(angle)

result = circuit(angle)
print(result)

# dev = qml.device('qiskit.ibmq.circuit_runner', wires=2, backend='ibmq_qasm_simulator', shots=8192)
token = '22f33f85e8f8144e9fa2b2dffa0b3a2fccf8396564670db4b52c0d00250263e47799f278c9584ebdb98964a655e806e4ff05fe73bd913f3496d3345a0abe43bd'
# dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_qasm_simulator', ibmqx_token=token)
dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_quito', ibmqx_token=token)

# dev = qml.device('qiskit.basicaer', wires=2)

# @qml.qnode(dev,interface='torch')
# def circuit(x):
#     qml.from_qiskit(qc)({theta: x})
#     return qml.expval(qml.PauliZ(0))
# qml.qnode

result = circuit(angle)
print(result)