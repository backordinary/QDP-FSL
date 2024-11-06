# https://github.com/KB00100100/paper-QSLT/blob/c1593a5c44d41f36ba4d0d4d88ca2e21242c0c9a/SVM_cancer.py
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
from qiskit_machine_learning.datasets import breast_cancer
from qiskit import IBMQ

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

#provider = IBMQ.enable_account('6d478a84688255f37f365f1d70bbdcacda8bb733daeaf5bf54d145ae08bbb96b9b0eabe8780900d7fe2475dc60862f50077fda8ceb34da08e991288bc1243ae2')

# number of qubits is equal to the number of features
num_qubits = 5
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
for i in range(1,100):
    #X_train, y_train, X_test, y_test = iris(
    #    training_size=32,
    #    test_size=8,
    #    n=dimension,
    #    plot_data=False,
    #    one_hot=False
    #)

    # X,y = make_classification(n_samples=200, n_features=5, n_classes=3, n_informative=3)
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    # # Now we standardize for gaussian around 0 with unit variance
    # std_scale = StandardScaler().fit(X_train)
    # X_train = std_scale.transform(X_train)
    # std_scale = StandardScaler().fit(X_test)
    # X_test = std_scale.transform(X_test)
    #
    # # Now reduce number of features to number of qubits
    # pca = PCA(n_components=num_qubits).fit(X_train)
    # X_train = pca.transform(X_train)
    # pca = PCA(n_components=num_qubits).fit(X_test)
    # X_test = pca.transform(X_test)
    #
    # # Scale to the range (-1,+1)
    # samples = np.append(X_train, X_test, axis=0)
    # minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    # X_train = minmax_scale.transform(X_train)
    # X_test = minmax_scale.transform(X_test)

    X_train, y_train, X_test, y_test = breast_cancer(training_size=28, test_size=12, n=num_qubits, plot_data=False, one_hot=False)
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


