# https://github.com/codigoscupom/QuantumAlgs/blob/8df3c7090622ea641fdadf1cba9e93f625e47ef7/VQC.py
from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data

seed = 1376
algorithm_globals.random_seed = seed

# Use ad hoc data set for training and test data
feature_dim = 2  # dimension of each data point
training_size = 20
test_size = 10

# training features, training labels, test features, test labels as np.array,
# one hot encoding for labels
training_features, training_labels, test_features, test_labels = \
    ad_hoc_data(
            training_size=training_size, test_size=test_size, n=feature_dim, gap=0.3)
#DETALHES SOBRE O AD HOC DATASET
#https://qiskit.org/documentation/machine-learning/tutorials/03_quantum_kernel.html?highlight=ad_hoc
print(training_features)
print(training_labels)
print(test_features)
print(test_labels)


feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement="linear")
ansatz = TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3)

#####HERE#########
vqc = VQC(feature_map=feature_map,
          ansatz=ansatz,
          optimizer=COBYLA(maxiter=100),
          quantum_instance=QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                           shots=1024,
                                           seed_simulator=seed,
                                           seed_transpiler=seed)
          )
vqc.fit(training_features, training_labels)

score = vqc.score(test_features, test_labels)
print(f"Testing accuracy: {score:0.2f}")