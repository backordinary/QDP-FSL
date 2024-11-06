# https://github.com/boyesjo/tma4500/blob/41c56c2cd29e3763832334701fd2c676f2cb4bc8/code/noise_fig/noise.py
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error
from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 1337

# %%
p_error = 0.003
deg_error = 1
shots = 1000

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(
    pauli_error([("X", p_error), ("I", 1 - p_error)]), "rx"
)

sim = AerSimulator(noise_model=noise_model, shots=shots)


def test_x_rotations(n: int):
    qc = QuantumCircuit(1, 1)
    for _ in range(n):
        qc.rx(np.pi + np.deg2rad(deg_error), 0)

    qc.measure(0, 0)
    return sim.run(qc).result().get_counts().get("0", 0)


# %%
l = [test_x_rotations(n) for n in range(1, 1000)]
l[::2] = [shots - i for i in l[::2]]
l = np.array(l) / shots

# %%
plt.plot(l)
plt.show()

# %%
pd.DataFrame(l, columns=["p"]).to_csv("noise.csv", index_label="n")
# %%
