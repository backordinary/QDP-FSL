# https://github.com/Linueks/QuantumComputing/blob/e85c410a8e9d47fd215b9dbd25a6c5b73daa1f6f/thesis/src/check_random_state_fidelity.py
import qiskit as qk
import numpy as np
import qiskit.opflow as opflow
from qiskit.quantum_info import random_unitary
import matplotlib.pyplot as plt


ket_zero = opflow.Zero
ket_one = opflow.One
input_state = ket_zero^ket_zero^ket_zero
final_state_target = ket_one^ket_one^ket_zero
target_state_matrix = final_state_target.to_matrix()

#print((random_unitary(2**3).data @ input_state.to_matrix()))
fidelity = np.zeros(10000)
for i in range(len(fidelity)):
    random_state = random_unitary(2**3).data @ input_state.to_matrix()
    fidelity[i] = ((final_state_target.to_matrix() @ (random_state.conj().T))* (random_state@(final_state_target.to_matrix().conj().T))).real
    #fidelity[i] = qk.quantum_info.state_fidelity(
    #            random_state,
    #            target_state_matrix,
    #        )


print(np.mean(fidelity), 2*np.std(fidelity))
plt.hist(fidelity)
plt.show()



# qiskit sample random_unitary
# qiskit state fidelity calculation
