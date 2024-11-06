# https://github.com/boyesjo/tma4500/blob/41c56c2cd29e3763832334701fd2c676f2cb4bc8/code/iris/qnn.py
import torch.nn as nn
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.utils import QuantumInstance
from qiskit_machine_learning import neural_networks
from qiskit_machine_learning.connectors import TorchConnector

NUM_QUBITS = 4


def parity(i: int) -> int:
    return bin(i).count("1") % 2


def last_bit(i: int) -> int:
    return int(bin(i)[-1])


def create_qnn(
    noisy: bool = False,
    interpret: str = "parity",
) -> neural_networks.NeuralNetwork:
    backend = Aer.get_backend("aer_simulator")
    shots = 1024

    if noisy:
        noise_model = NoiseModel.from_backend(FakeMontreal())
        qi = QuantumInstance(
            backend=backend,
            noise_model=noise_model,
            shots=shots,
        )
    else:
        qi = QuantumInstance(backend=backend, shots=shots)

    if interpret == "parity":
        interpret_func = parity
    elif interpret == "last_bit":
        interpret_func = last_bit
    else:
        raise ValueError(f"Unknown interpretation: {interpret}")

    fm = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=2)
    ansatz = RealAmplitudes(num_qubits=NUM_QUBITS, reps=1)
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(fm, range(NUM_QUBITS))
    qc.append(ansatz, range(NUM_QUBITS))

    qnn = neural_networks.CircuitQNN(
        qc,
        input_params=fm.parameters,
        weight_params=ansatz.parameters,
        interpret=interpret_func,
        output_shape=2,
        quantum_instance=qi,
    )

    return qnn


class QNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.qnn = TorchConnector(create_qnn(**kwargs))

    def forward(self, x):
        x = self.qnn.forward(x)
        return x


if __name__ == "__main__":

    ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=2).decompose().draw(
        output="latex", filename="fm.pdf"
    )

    RealAmplitudes(num_qubits=NUM_QUBITS, reps=1).decompose().draw(
        output="latex", filename="ansatz.pdf"
    )
