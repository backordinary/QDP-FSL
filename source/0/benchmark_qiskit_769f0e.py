# https://github.com/donghufeng/mq_benchmark/blob/b778c79013b834d79cdd37be6f4b3aa0ff0b8e1c/qnn/benchmark_qiskit.py
from gc import callbacks
import time
import numpy as np
from data_generator import circle_data_point_generator
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from tran_mcircuit import to_qiskit
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli
from qiskit_machine_learning.connectors import TorchConnector
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Adam
from mindquantum.core import Circuit as mcircuit
from qiskit.opflow import AerPauliExpectation


def bench(n_qubits, epoch, batch, n_samples):

    Ntrain = n_samples
    Ntest = 100
    gap = 0.01

    # 生成数据集
    train_features, train_labels, test_features, test_labels = circle_data_point_generator(
        Ntrain, Ntest, gap, n_qubits, 1
    )

    # 生成encoder
    mencoder = mcircuit()
    for i in range(n_qubits):
        mencoder.ry(f"alpha_{i}", i)

    encoder = to_qiskit(mencoder)

    # 生成ansatz
    mansatz = mcircuit()
    for i in range(n_qubits):
        mansatz.rz(f"beta_1_{i}", i)
        mansatz.ry(f"beta_2_{i}", i)
        mansatz.rz(f"beta_3_{i}", i)

    for i in range(n_qubits - 1):
        mansatz.x(i + 1, i)
    mansatz.x(0, n_qubits - 1)

    for i in range(n_qubits):
        mansatz.ry(f"beta_4_{i}", i)

    ansatz = to_qiskit(mansatz)

    observable = PauliOp(Pauli("Z"), 1.0)
    qi_sv = QuantumInstance(Aer.get_backend("statevector_simulator"))

    # 生成网络
    qnn = TwoLayerQNN(
        n_qubits,
        feature_map=encoder,
        input_gradients=True,
        ansatz=ansatz,
        observable=observable,
        exp_val=AerPauliExpectation(),
        quantum_instance=qi_sv,
    )


    initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn.num_weights) - 1)

    # 生成模型
    model1 = TorchConnector(qnn, initial_weights=initial_weights)
    optimizer = Adam(model1.parameters(), lr=0.1)
    f_loss = MSELoss(reduction="mean")
    model1.train()

    print("qiskit: start training")
    # 训练模型
    start_time = time.time()
    for i in range(epoch):
        for j in range(n_samples//batch):
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = model1(Tensor(train_features[j*batch:j*batch+batch]))  # Forward pass
            loss = f_loss(output, Tensor(train_labels[j*batch:j*batch+batch]))  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
    end_time = time.time()
    print("qiskit: finish")
    return end_time - start_time

    '''
    # Evaluate model and compute accuracy
    y_predict = []
    for i in range(len(test_features)):
        output = model1(Tensor(test_features[i]))
        y_predict.append(np.sign(output.detach().numpy())[0])

    print("Accuracy:", sum(y_predict == test_labels) / len(test_labels))
    '''

if __name__=='__main__':
    res = bench(4, 5, 20, 200)
    print(res)