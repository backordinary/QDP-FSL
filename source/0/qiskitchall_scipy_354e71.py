# https://github.com/ryuNagai/QML/blob/54fc9e51e4fea7be8b8d4e51686a2496b81089c8/TFq/trial/QISKITchall_scipy.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit.tools.visualization import plot_state_city

from qiskit.providers.aer import StatevectorSimulator
from scipy.optimize import minimize

def get_cirq(params):
    qc = QuantumCircuit(4)
    qc.rx(params[0], 0)
    qc.ry(params[1], 0)
    qc.rz(params[2], 0)
    qc.rx(params[3], 1)
    qc.ry(params[4], 1)
    qc.rz(params[5], 1)
    qc.rx(params[6], 2)
    qc.ry(params[7], 2)
    qc.rz(params[8], 2)
    qc.rx(params[9], 3)
    qc.ry(params[10], 3)
    qc.rz(params[11], 3)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    return qc

simulator = Aer.get_backend('statevector_simulator')
# Execute and get counts

target = np.array([
    -0.21338835+0.33838835j, -0.14016504-0.08838835j,
    0.21338835-0.08838835j, 0.03661165+0.08838835j,
    0.08838835-0.03661165j, -0.08838835-0.21338835j,
    -0.08838835+0.14016504j,  0.33838835+0.21338835j,
    0.21338835-0.08838835j, 0.03661165+0.08838835j,
    0.39016504+0.08838835j, -0.03661165+0.16161165j,
    0.16161165+0.03661165j, 0.08838835-0.39016504j,
    0.08838835-0.03661165j, -0.08838835-0.21338835j])

def loss(param):
    qc = get_cirq(param)
    result = execute(qc, simulator).result()
    statevector = result.get_statevector(qc)
    diff = np.sum(np.square(np.abs(statevector - target)))
    return diff

def get_state(param):
    qc = get_cirq(param)
    result = execute(qc, simulator).result()
    statevector = result.get_statevector(qc)
    return statevector

loss_list = []
params_list = []
state_list = []
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))
params = np.random.rand(12)
res = minimize(loss, params, method='Powell')
loss_list.append(loss(res.x))
params_list.append(res.x)
state_list.append(get_state(res.x))

print(min(loss_list))