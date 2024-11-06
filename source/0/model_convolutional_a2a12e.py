# https://github.com/Giermek222/BFQCA-Benchmarking-For-Quantum-Classification-Algorithms-/blob/fd8eea80bc65ad1092bfa0c51bc73a2b52368382/BFQCA_Cowskit/library/cowskit/models/model_convolutional.py
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.algorithms.optimizers.adam_amsgrad import ADAM
from qiskit.quantum_info import SparsePauliOp

from cowskit.models.model import Model
from cowskit.datasets.dataset import Dataset
from cowskit.utils import save_circuit_drawing
class ConvolutionalModel(Model):
    def __init__(self, dataset: Dataset = None) -> None:
        Model.__init__(self, dataset)
        self.EPOCHS = 2
        
    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        if self.get_output_size() != 1:
            raise Exception("Convolutional model accepts only binary classification data")

        Y = Y.flatten()
        self.build_circuit()
        save_circuit_drawing(self.circuit.decompose(), "8bit_qcnn")
        self.classifier.fit(X, Y)

    def predict(self, value: np.ndarray):
        return self.classifier.predict(value)
            
    def build_circuit(self):
        in_size = self.get_input_size()

        self.feature_map = ZFeatureMap(in_size)
        self.instruction_set = QuantumCircuit(in_size)

        layer_id = 0
        sources = list(range(in_size))
        all_qubits = list(range(in_size))

        if in_size == 1:
            self.instruction_set.compose(self.conv_layer(sources, layer_id), all_qubits, inplace=True)
        else:
            while len(sources) != 1:
                compose_targets = list(range(in_size - len(sources), in_size))
                pooling_split = len(sources) // 2
                pooling_sources = sources[:pooling_split]
                pooling_sinks = sources[pooling_split:]
                self.instruction_set.compose(self.conv_layer(sources, layer_id), compose_targets, inplace=True)
                self.instruction_set.compose(self.pool_layer(pooling_sources, pooling_sinks, layer_id), compose_targets, inplace=True)
                sources = pooling_sources
                layer_id += 1

        self.circuit = QuantumCircuit(in_size)
        self.circuit.compose(self.feature_map,     all_qubits, inplace=True)
        self.circuit.compose(self.instruction_set, all_qubits, inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * (in_size - 1), 1)])

        self.network = EstimatorQNN(
            circuit=self.circuit.decompose(),
            input_params=self.feature_map.parameters,
            weight_params=self.instruction_set.parameters,
            observables=observable
        )

        self.classifier = NeuralNetworkClassifier(
            self.network,
            optimizer=ADAM(maxiter=self.EPOCHS),
        )


    def conv_layer(self, sources, id):
        def conv_circuit(params):
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)
            target.cx(1, 0)
            target.rz(np.pi / 2, 0)
            return target

        num_qubits = len(sources)
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(f"w{str(id + 1)}", length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def pool_layer(self, sources, sinks, id):
        def pool_circuit(params):
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)

            return target

        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(f"p{str(id + 1)}", length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc
