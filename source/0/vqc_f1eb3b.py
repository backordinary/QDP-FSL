# https://github.com/boyesjo/tma4500/blob/41c56c2cd29e3763832334701fd2c676f2cb4bc8/code/iris_multiclass/vqc.py
# %%
import numpy as np
import pandas as pd
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms import optimizers
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning import neural_networks
from qiskit_machine_learning.algorithms.classifiers import (
    VQC,
    NeuralNetworkClassifier,
)
from qiskit_machine_learning.connectors import TorchConnector
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize

NUM_QUBITS = 4


# %%
def load_data() -> tuple[np.ndarray, np.ndarray]:
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )

    x = normalize(df.drop("target", axis=1).values)
    y = df["target"].values.astype(int)

    # shuffle
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]

    # one hot encode y
    y = np.eye(3)[y]
    return x, y


x, y = load_data()


def interpret(i: int) -> tuple[int, int, int]:
    ret = (i & 1, (i >> 1) & 1, (i >> 2) & 1)
    return ret


# create global iteration counter
global it
it = 0


def callback(_, __):
    global it
    it += 1
    print(f"iteration: {it}")


qi = QuantumInstance(backend=Aer.get_backend("aer_simulator"))

fm = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=2)

ansatz = RealAmplitudes(num_qubits=NUM_QUBITS, reps=1)

qc = QuantumCircuit(NUM_QUBITS)
qc.append(fm, range(NUM_QUBITS))
qc.append(ansatz, range(NUM_QUBITS))

qnn = VQC(
    num_qubits=NUM_QUBITS,
    quantum_instance=qi,
    feature_map=fm,
    ansatz=ansatz,
    callback=callback,
    optimizer=optimizers.ADAM(maxiter=100),
)


# %%
qnn.fit(x, y)


# %%
# get accuracy
y_pred = qnn.predict(x)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y, axis=1)
acc = np.sum(y_pred == y_true) / len(y_true)
print(f"accuracy: {acc}")
# %%
