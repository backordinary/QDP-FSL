# https://github.com/KB00100100/paper-QSLT/blob/c1593a5c44d41f36ba4d0d4d88ca2e21242c0c9a/SVM_wine.py
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from qiskit import BasicAer
from qiskit.circuit.library import ZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC,PegasosQSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import wine

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


# number of qubits is equal to the number of features
num_qubits = 7
# # number of steps performed during the training procedure
# tau = 100
# # regularization parameter
# C = 1000

algorithm_globals.random_seed = 12345

backend = QuantumInstance(
    BasicAer.get_backend("statevector_simulator"),
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)
#backend = provider.get_backend('ibmq_bogota')

feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)


total_svc = 0
total_qsvc = 0
for i in range(1,101):
    X_train, y_train, X_test, y_test = wine(training_size=28, test_size=12, n=num_qubits, plot_data=False, one_hot=False)
    #print(y_train)

    ## ------- for SVC ---------##
    svc = SVC()
    svc.fit(X_train, y_train)
    svc_score = svc.score(X_test, y_test)
    total_svc += svc_score

    ## ------- for QSVC---------##
    qsvc = QSVC(quantum_kernel=qkernel)
    # training
    qsvc.fit(X_train, y_train)
    # testing
    qsvc_score = qsvc.score(X_test, y_test)
    total_qsvc += qsvc_score

    print(f"PRNGs: {svc_score}"
          f"\tQRNGs: {qsvc_score}")

print(f"avg PRNGs: {total_svc/100}"
      f"\tavg QRNGs: {total_qsvc/100}")


