# https://github.com/dreaming-panda/quantum-ml/blob/ca14e6563816bc72b81ea87985d3f5d3e798e243/problem.py
from config import QuantumConfig
from data import QuantumDataSet
from model import QuantumModel
from optimizer import GroverOptimizer, QuantumOptimizer
from qiskit import QuantumCircuit
class XORDataset(QuantumDataSet):
    def __init__(self, features: list[list], label: list, qc: QuantumCircuit) -> None:
        super().__init__(features, label, qc)
    def encode(self) -> None:
        for q in self.features[0]:
            self.qc.h(q)
        self.qc.barrier()
        self.qc.x(self.label[0])
        self.qc.ccx(self.features[0][0], self.features[0][1], self.label[0])
        self.qc.x(self.features[0][0])
        self.qc.x(self.features[0][1])
        self.qc.ccx(self.features[0][0], self.features[0][1], self.label[0])
        self.qc.x(self.features[0][0])
        self.qc.x(self.features[0][1])
        self.qc.barrier()

        for q in self.features[1]:
            self.qc.h(q)
        self.qc.barrier()
        self.qc.x(self.label[1])
        self.qc.ccx(self.features[1][0], self.features[1][1], self.label[1])
        self.qc.x(self.features[1][0])
        self.qc.x(self.features[1][1])
        self.qc.ccx(self.features[1][0], self.features[1][1], self.label[1])
        self.qc.x(self.features[1][0])
        self.qc.x(self.features[1][1])
        self.qc.barrier()

    def decode(self) -> None:
        self.qc.x(self.features[1][1])
        self.qc.x(self.features[1][0])
        self.qc.ccx(self.features[1][0], self.features[1][1], self.label[1])
        self.qc.x(self.features[1][1])
        self.qc.x(self.features[1][0])
        self.qc.ccx(self.features[1][0], self.features[1][1], self.label[1])
        self.qc.x(self.label[1])
        self.qc.barrier()
        for q in self.features[1]:
            self.qc.h(q)
        self.qc.barrier()
        self.qc.x(self.features[0][1])
        self.qc.x(self.features[0][0])
        self.qc.ccx(self.features[0][0], self.features[0][1], self.label[0])
        self.qc.x(self.features[0][1])
        self.qc.x(self.features[0][0])
        self.qc.ccx(self.features[0][0], self.features[0][1], self.label[0])
        self.qc.x(self.label[0])
        self.qc.barrier()
        for q in self.features[0]:
            self.qc.h(q)
class ANDXORModel(QuantumModel):
    def __init__(self, config: QuantumConfig) -> None:
        super().__init__(config)
    def compute(self) -> None:
        self.qc.ccx(self.weights[0], self.features[0][0], self.outputs[0])
        self.qc.ccx(self.weights[1], self.features[0][1], self.outputs[0])
        self.qc.barrier()
        self.qc.ccx(self.weights[0], self.features[1][0], self.outputs[1])
        self.qc.ccx(self.weights[1], self.features[1][1], self.outputs[1])
        self.qc.barrier()
    def uncompute(self) -> None:
        self.qc.ccx(self.weights[1], self.features[1][1], self.outputs[1])
        self.qc.ccx(self.weights[0], self.features[1][0], self.outputs[1])
        self.qc.barrier()
        self.qc.ccx(self.weights[1], self.features[0][1], self.outputs[0])
        self.qc.ccx(self.weights[0], self.features[0][0], self.outputs[0])
        self.qc.barrier()
if __name__ == '__main__':
    n = 10
    m = 10
    qc = QuantumCircuit(n, m)
    weights = [0, 1]
    features = [[2, 3], [6, 7]]
    labels = [4, 8]
    outputs = [5, 9]
    config = QuantumConfig(weights=weights, features=features, labels=labels, output=outputs, qc=qc, classical_bits=list(range(m)))
    dataset = XORDataset(features=features, label=labels,qc=qc)
    model = ANDXORModel(config=config)
    optimier = GroverOptimizer(config=config, data=dataset, model=model, iter=9)
    optimier.optimize()
    counts = optimier.execute(draw=False)
    print(counts)
