# https://github.com/Linueks/QuantumComputing/blob/e85c410a8e9d47fd215b9dbd25a6c5b73daa1f6f/thesis/src/bloch_sphere.py
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_vector


plot_bloch_vector(
    [0,0,0],
    title="New Bloch Sphere",
)
plt.show()
