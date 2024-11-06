# https://github.com/lucasgrisiq/quantum-classifier/blob/c7e0a021383fec9fe2243bfdb0d78c6e10049ab5/perceptron.py
import qiskit
from qiskit.algorithms.optimizers import SPSA
from noisyopt import minimizeSPSA
import numpy as np
 
 
class QPerceptron():
 
    def __init__(self, n):
        self.n = n
        self.qstates = qiskit.QuantumRegister(n, 'qr')
        self.sim = qiskit.Aer.get_backend('qasm_simulator')
 
    def _nBin(self, x):
        return (''.join('0' for _ in range(self.n)) + bin(x)[2:])[-self.n:]
 
    def set_inputs(self, inputs):
        self.theta = inputs
 
    def set_weights(self, weights):
        self.phi = weights
 
    def _make_ui(self):
        q = qiskit.QuantumRegister(self.n, 'qr')
        ui = qiskit.QuantumCircuit(q)
        ui.h(q)
 
        prev_step = self._nBin(0)
        for s in range(1, np.power(2, self.n)):
            step = self._nBin(s)
            for i, state in enumerate(step):
                if state != prev_step[i]:
                    ui.x(q[i])
 
            th = self.theta[s] - self.theta[0]
            ui.mcrz(lam=th, q_controls=q[:-1], q_target=q[-1])
 
            prev_step = step
 
        return ui.to_gate()
 
    def _make_uw(self):
        q = qiskit.QuantumRegister(self.n, 'qr')
        uw = qiskit.QuantumCircuit(q)
 
        prev_step = self._nBin(0)
        for s in range(1, np.power(2, self.n)):
            step = self._nBin(s)
            for i, state in enumerate(step):
                if state != prev_step[i]:
                    uw.x(q[i])
 
            th = self.phi[s] - self.phi[0]
            uw.mcrz(lam=th, q_controls=q[:-1], q_target=q[-1])
 
            prev_step = step
 
        uw.h(q)
        uw.x(q)
 
        return uw.to_gate()
 
    def make_ui_gates(self, inputs: list):
        self.ui_gates = []
        for inpt in inputs:
            self.set_inputs(inpt)
            gate = self._make_ui()
            self.ui_gates.append(gate)
 
    def make_circuit(self, ui: qiskit.circuit.Gate, uw: qiskit.circuit.Gate):
        ansilla = qiskit.QuantumRegister(1, 'ansl')
        meas = qiskit.ClassicalRegister(1, 'cr0')
        circuit = qiskit.QuantumCircuit(self.qstates, ansilla, meas)
 
        circuit.append(ui, self.qstates)
        circuit.append(uw, self.qstates)
 
        circuit.mcx(self.qstates, ansilla)
        circuit.measure(ansilla, meas)
 
        return circuit
 
    def run(self, index: int, uw: qiskit.circuit.Gate, threshold: float = 0.5):
 
        ui = self.ui_gates[index]
        self.circuit = self.make_circuit(ui=ui, uw=uw)
 
        result = qiskit.execute(self.circuit, self.sim, shots=10).result().get_counts()
        resp = result.get('1', 0) / 10
 
        if resp > threshold:
            return 1, resp
        else:
            return 0, resp
 
    def cost_function(self, w: list):
        # make uw gate
        self.set_weights(w)
        uw = self._make_uw()
 
        # init calculation
        cost = 0
        for i in range(len(self.input_states)):
            y, _ = self.run(i, uw)
            cost += (self.labels[i] - y) ** 2
 
        cost /= len(self.input_states)
        self.cost_array.append(cost)
 
        return cost
 
    def callback_spsa(self, xk):
        print('\t## Iter: {} | Custo: {}'.format(len(self.cost_array), self.cost_array[-1]))
 
    def fit(self, X: list, y: list, niter: int = 300, learning_rate: float = 0.2):
        # Set training data
        if X is not None and y is not None:
            self.input_states = X
            self.labels = y
 
        print('# No of inputs: {}'.format(len(self.input_states)))
        print('# Input size: {}'.format(len(self.input_states[0])))
        print('# Labels: {}'.format(', '.join(map(str, np.unique(self.labels)))))
 
        # make ui gates
        self.make_ui_gates(self.input_states)
        print('# Inputs processed')
 
        size = len(self.input_states[0])
        x0 = np.random.uniform(0, np.pi/2, size)
        self.cost_array = []
 
        print('Starting optimization')
        print('# Iterations: {}'.format(niter))
        print('# Learning rate: {}'.format(learning_rate))
 
        res = minimizeSPSA(
            func=self.cost_function,
            x0=x0,
            bounds=[(0, np.pi/2)]*size,
            niter=niter,
            paired=False,   # sem seed
            c=0.1, # aleatoriedade  
            a=learning_rate,
            callback=self.callback_spsa
        )
 
        print('Optimization finished')
        print('# Final weights: {}'.format(res.x))
        print('# Final cost: {}'.format(res.fun))
        print('# Cost array: {}'.format(self.cost_array))
        print(res)

        self.set_weights(res.x)
        return res
    
    def predict(self, X: list, threshold: float = 0.5):
        self.input_states = X
        labels = []
        self.make_ui_gates(self.input_states)
        uw = self._make_uw()
        for i in range(len(self.input_states)):
            y, resp = self.run(i, uw, threshold=threshold)
            labels.append((y, resp))
        return labels