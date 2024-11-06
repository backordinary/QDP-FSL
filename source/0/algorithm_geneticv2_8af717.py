# https://github.com/Giermek222/BFQCA-Benchmarking-For-Quantum-Classification-Algorithms-/blob/fd8eea80bc65ad1092bfa0c51bc73a2b52368382/BFQCA_Cowskit/library/cowskit/algorithms/algorithm_geneticV2.py
import numpy as np

from typing import List, Tuple
from cowskit.algorithms import Algorithm
from cowskit.datasets import Dataset
from cowskit.utils import get_shape_size, compute_binary_crossentropy_loss, compute_categorical_crossentropy_loss
from cowskit.utils import bin_to_float, float_to_bin, sigmoid, softmax, relu, tanh, one_hot, sign
from cowskit.utils import fast_binary_accuracy, fast_categorical_accuracy

from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

class MutatedChampionsGeneticAlgorithm(Algorithm):
    def __init__(self, dataset: Dataset = None) -> None:
        Algorithm.__init__(self, dataset)
        self.PI = 3.1416
        self.float_precision = 8
        self.epochs = 100
        self.population = 256
        self.best_performers_count = 8
        self.deep_layers = 1
        self.angle_nudge_radians = self.PI/3
        self.trained_network:List[np.ndarray] = []

        assert(self.population % self.best_performers_count == 0)
        assert(self.float_precision % 8 == 0)

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:

        n_features = get_shape_size(X)
        n_classes = get_shape_size(Y)

        networks = self.construct_starting_networks(n_features, n_classes)
        best_performers_priority_queue: List[Tuple[float, List[np.ndarray]]] = []
        best_performers: List[List[np.ndarray]] = []

        for epoch in range(self.epochs):
            best_performers_priority_queue.clear()
            best_performers.clear()

            # Evaluate all networks
            for network in networks:
                result = X.copy()
                for deep_layer in range(self.deep_layers):
                    result = result @ network[deep_layer]
                    if n_classes == 1:
                        result = tanh(result)
                    else:
                        result = sigmoid(result)
                
                result = result @ network[self.deep_layers]

                if n_classes == 1:
                    result = tanh(result)
                    loss = compute_binary_crossentropy_loss(Y, result)
                    result = sign(result)
                    acc = fast_binary_accuracy(Y, result)
                else:
                    result = softmax(result)
                    loss = compute_categorical_crossentropy_loss(Y, result)
                    result = one_hot(result)
                    acc = fast_categorical_accuracy(Y, result)
                
                best_performers_priority_queue.append((acc, loss, network))

            # Get best performers
            best_performers_priority_queue.sort(key=lambda x:x[0], reverse=True)
            best_performers_priority_queue = best_performers_priority_queue[:self.best_performers_count]
            print(f"Epoch: {epoch} Est.Accuracy: {best_performers_priority_queue[0][0]} Loss: {best_performers_priority_queue[0][1]}")
            best_performers = list(map( lambda x:x[2], best_performers_priority_queue))
            
            if epoch == self.epochs - 1:
                break
            
            # construct new population
            networks = self.construct_new_networks(best_performers, n_features, n_classes)
        
        self.trained_network = best_performers[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y = X.copy()

        n_classes = self.trained_network[self.deep_layers].shape[1]

        for deep_layer in range(self.deep_layers):
            Y = Y @ self.trained_network[deep_layer]
            if n_classes == 1:
                Y = tanh(Y)
            else:
                Y = sigmoid(Y)

        Y = Y @ self.trained_network[self.deep_layers]

        if n_classes == 1:
            Y = tanh(Y)
            Y = sign(Y)
        else:
            Y = softmax(Y)
            Y = one_hot(Y)

        return Y

    def construct_new_networks(self, best_performers:List[List[np.ndarray]], neruons_per_layer:int, output_classes:int):


        new_networks = best_performers.copy()
        children_amount = self.population // self.best_performers_count - 1
        # get probabilities of each float-bit to be equal to 1
        for network_idx in range(self.best_performers_count):
            mutated_networks = self.construct_empty_networks(children_amount, neruons_per_layer, output_classes)
            for layer_idx in range(self.deep_layers+1):
                layer = best_performers[network_idx][layer_idx]
                shape = layer.shape
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        binary = float_to_bin(layer[x][y], self.float_precision)
                        mutated_float_weights = self.generate_quantum_population(children_amount, binary)
                        for i,f in enumerate(mutated_float_weights):
                            mutated_networks[i][layer_idx][x][y] = f
            new_networks.extend(mutated_networks)

        return new_networks

    def generate_quantum_population(self, amount: int, binary: str):
        backend = Aer.get_backend('aer_simulator')
        results = np.zeros((amount), dtype=np.float32)

        def get_result(idx:int, binary: str, backend:AerSimulator):
            quantum_circuit = QuantumCircuit()
            qr = QuantumRegister(8)
            cr = ClassicalRegister(8)
            quantum_circuit.add_register(qr)
            quantum_circuit.add_register(cr)
            quantum_circuit.h(qr[0])
            rotation_angle = 0 if binary[idx+0] == '0' else self.PI
            quantum_circuit.rx(rotation_angle, qr[0])
            quantum_circuit.rx(self.angle_nudge_radians, qr[0])
            quantum_circuit.ry(self.angle_nudge_radians, qr[0])
            quantum_circuit.rz(self.angle_nudge_radians, qr[0])
            quantum_circuit.measure(qr[0],cr[0])
            quantum_circuit.h(qr[1])
            rotation_angle = 0 if binary[idx+1] == '0' else self.PI
            quantum_circuit.rx(rotation_angle, qr[1])
            quantum_circuit.rx(self.angle_nudge_radians, qr[1])
            quantum_circuit.ry(self.angle_nudge_radians, qr[1])
            quantum_circuit.rz(self.angle_nudge_radians, qr[1])
            quantum_circuit.measure(qr[1],cr[1])
            quantum_circuit.h(qr[2])
            rotation_angle = 0 if binary[idx+2] == '0' else self.PI
            quantum_circuit.rx(rotation_angle, qr[2])
            quantum_circuit.rx(self.angle_nudge_radians, qr[2])
            quantum_circuit.ry(self.angle_nudge_radians, qr[2])
            quantum_circuit.rz(self.angle_nudge_radians, qr[2])
            quantum_circuit.measure(qr[2],cr[2])
            quantum_circuit.h(qr[3])
            rotation_angle = 0 if binary[idx+3] == '0' else self.PI
            quantum_circuit.rx(rotation_angle, qr[3])
            quantum_circuit.rx(self.angle_nudge_radians, qr[3])
            quantum_circuit.ry(self.angle_nudge_radians, qr[3])
            quantum_circuit.rz(self.angle_nudge_radians, qr[3])
            quantum_circuit.measure(qr[3],cr[3])
            quantum_circuit.h(qr[4])
            rotation_angle = 0 if binary[idx+4] == '0' else self.PI
            quantum_circuit.rx(rotation_angle, qr[4])
            quantum_circuit.rx(self.angle_nudge_radians, qr[4])
            quantum_circuit.ry(self.angle_nudge_radians, qr[4])
            quantum_circuit.rz(self.angle_nudge_radians, qr[4])
            quantum_circuit.measure(qr[4],cr[4])
            quantum_circuit.h(qr[5])
            rotation_angle = 0 if binary[idx+5] == '0' else self.PI
            quantum_circuit.rx(rotation_angle, qr[5])
            quantum_circuit.rx(self.angle_nudge_radians, qr[5])
            quantum_circuit.ry(self.angle_nudge_radians, qr[5])
            quantum_circuit.rz(self.angle_nudge_radians, qr[5])
            quantum_circuit.measure(qr[5],cr[5])
            quantum_circuit.h(qr[6])
            rotation_angle = 0 if binary[idx+6] == '0' else self.PI
            quantum_circuit.rx(rotation_angle, qr[6])
            quantum_circuit.rx(self.angle_nudge_radians, qr[6])
            quantum_circuit.ry(self.angle_nudge_radians, qr[6])
            quantum_circuit.rz(self.angle_nudge_radians, qr[6])
            quantum_circuit.measure(qr[6],cr[6])
            quantum_circuit.h(qr[7])
            rotation_angle = 0 if binary[idx+7] == '0' else self.PI
            quantum_circuit.rx(rotation_angle, qr[7])
            quantum_circuit.rx(self.angle_nudge_radians, qr[7])
            quantum_circuit.ry(self.angle_nudge_radians, qr[7])
            quantum_circuit.rz(self.angle_nudge_radians, qr[7])
            quantum_circuit.measure(qr[7],cr[7])

            job = backend.run(quantum_circuit, shots = amount, memory=True)
            return job.result().get_memory()

        float_bits = [] 
        for i in range(self.float_precision // 8):
            float_bits.append(get_result(i * 8,  binary, backend))
        float_bits = list(zip(*float_bits))
        for i in range(amount):
            binary_float = "".join(float_bits[i])
            generated_float = bin_to_float(binary_float, self.float_precision)
            results[i] = generated_float
            
        return results

    def construct_starting_networks(self, neruons_per_layer:int, output_classes:int) -> List[List[np.ndarray]]:
        starting_networks: List[List[np.ndarray]] = [[]]*self.population
        for i in range(self.population):
            network = [0]*(self.deep_layers+1)
            for d in range(self.deep_layers):
                network[d] = np.random.rand(neruons_per_layer, neruons_per_layer).astype(np.float32)
            network[self.deep_layers] = np.random.rand(neruons_per_layer, output_classes).astype(np.float32)
            starting_networks[i] = network

        return starting_networks

    def construct_empty_networks(self, amount:int, neruons_per_layer:int, output_classes:int) -> List[List[np.ndarray]]:
        construct_empty_networks: List[List[np.ndarray]] = [[]]*amount
        for i in range(amount):
            network = [0]*(self.deep_layers+1)
            for d in range(self.deep_layers):
                network[d] = np.zeros((neruons_per_layer, neruons_per_layer))
            network[self.deep_layers] = np.zeros((neruons_per_layer, output_classes))
            construct_empty_networks[i] = network

        return construct_empty_networks




