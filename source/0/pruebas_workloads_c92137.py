# https://github.com/yessichibii/Covalent/blob/3b0dee022170c3e8d382da236ac905c4a23afb2e/Pruebas_workloads.py
import covalent as ct
import pennylane as qml
import numpy as np
from math import pi as ppi
from qiskit import IBMQ
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

IBMQ.load_account()
#from qiskit import Aer, BasicAer

###### INICIA LATTICE DE FEATUREMAPS #######

@ct.electron
def return_dev(backend, n):
    dev = qml.device(backend, wires = n)
    return dev

###### CIRCUITOS ##############

@ct.electron
def ZZ_probs(x):
    features = x.shape[0]
    for i in range(features):
        qml.Hadamard(wires = i)
        qml.RZ(2*x[i], wires = i)

    for j in range(features - 1):
        for i in range(features - 1):
            qml.CNOT(wires = [i, i+1])
            qml.RZ(2*(np.pi - x[i])*(np.pi - x[i + 1]), wires = i + 1)
            qml.CNOT(wires = [i, i + 1])

        if j < features - 2:
            for i in range(features):
                qml.Hadamard(wires = i)
                qml.RZ(2*x[i], wires = i)

    return qml

@ct.electron
def PZ(x):
    features = x.shape[0]
    for i in range(features):
        qml.Hadamard(wires = i)
        qml.RZ(2*x[i], wires = i)

    return qml

@ct.electron
def PX(x):
    features = x.shape[0]
    for i in range(features):
        qml.Hadamard(wires = i)
        qml.Hadamard(wires = i)
        qml.RZ(2*x[i], wires = i)
        qml.Hadamard(wires = i)
    return qml

@ct.electron
def PY(x):
    features = x.shape[0]
    for i in range(features):
        qml.Hadamard(wires = i)
        qml.RX(ppi/2, wires = i)
        qml.RZ(2*x[i], wires = i)
        qml.RX(-ppi/2, wires = i)
    return qml


##### PROBABILIDADES  #########

@ct.electron
def ZZfeatureMap(x, y):
    features = x.shape[0]
    ZZ_probs(x)
    qml.adjoint(ZZ_probs)(y)
    return qml.probs(wires = range(features))

@ct.electron
def PaulifeatureMapZ(x, y):
    features = x.shape[0]
    PZ(x)
    qml.adjoint(PZ)(y)
    return qml.probs(wires = range(features))

@ct.electron
def PauliFeatureMapX(x, y):
    features = x.shape[0]
    PX(x)
    qml.adjoint(PX)(y)
    return qml.probs(wires = range(features))

@ct.electron
def PauliFeatureMapY(x, y):
    features = x.shape[0]
    PY(x)
    qml.adjoint(PY)(y)
    return qml.probs(wires = range(features))

##### ENVIO DE PROBABILIDADES CON BACKEND #########

@ct.electron
def ZZ(x, y, backend):
    num_wires = x.shape[0]
    dev_z = return_dev(backend, num_wires)
    qml_z = qml.QNode(ZZfeatureMap, dev_z)
    resultado = qml_z(x, y)
    return resultado

@ct.electron
def PauliX(x, y, backend):
    num_wires = x.shape[0]
    dev_px = return_dev(backend, num_wires)
    qml_px = qml.QNode(PauliFeatureMapX, dev_px)
    resultado = qml_px(x, y)
    return resultado

@ct.electron
def PauliY(x, y, backend):
    num_wires = x.shape[0]
    dev_py = return_dev(backend, num_wires)
    qml_py = qml.QNode(PauliFeatureMapY, dev_py)
    resultado = qml_py(x, y)
    return resultado

@ct.electron
def PauliZ(x, y, backend):
    num_wires = x.shape[0]
    dev_pz = return_dev(backend, num_wires)
    qml_pz = qml.QNode(PaulifeatureMapZ, dev_pz)
    resultado = qml_pz(x, y)
    return resultado

#### CIRCUITO COMPLETO ########

@ct.electron
@ct.lattice
def circuitoFM(x, y, backend):
    zz = ZZ(x, y, backend)
    py = PauliY(x, y, backend)
    px = PauliX(x, y, backend)
    pz = PauliZ(x, y, backend)
    lista_probas = [zz, py, px, pz]
    return lista_probas

###### TERMINA LATTICE DE FEATUREMAPS #######

###### INICIA LATTICE DE QSVM FUNCIONES ###############


def get_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # Sólo 2 features
    y = iris.target
    return X, y

def split_train_test_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test

##### LATTICES CHIDOS #######

@ct.electron
@ct.lattice
def parametros_workflow(X1, X2, backend, len_probas):
    mapa = ['zz', 'py', 'px', 'pz']
    rango = len_probas
    gram_matrix = np.zeros((len(mapa), X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            listaMapa = circuitoFM(X1,X2, backend)
            for ele in range(len_probas):
                gram_matrix[ele][i, j] = listaMapa[ele]

    return gram_matrix

@ct.lattice
def workflow(x_train, x_test, y_train, y_test, backend, rango):
    clf = {}
    matrix = parametros_workflow(x_test, x_train, backend, rango)
    for ele in range(rango):
        clf[ele] = svm.SVC(kernel="precomputed")
        clf[ele].fit(matrix[ele], y_train)

    testMatrix = parametros_workflow(x_test, x_train, backend)
    qsvc = {}
    for ele in range(rango):
        qsvc[ele] = clf[ele].predict()

    success = np.zeros(rango)

    for ele in range(rango):
        for i in range(len(y_test)):
            if qsvc[ele][i] == y_test[i]:
                success[ele] += 1
    promedio_success = np.round(success/len(y_test) * 100, 2)

    return promedio_success


results = []



#backends = ["qiskit.aer", "qiskit.basicaer", "qiskit.ibmq", 'qasm_simulator','qasm_simulator_py', 'statevector_simulator',
# 'statevector_simulator_py', 'unitary_simulator', 'clifford_simulator']

#backends = ["qiskit.aer", "qiskit.basicaer", 'qiskit.ibmq', 'qiskit.ibmq.circuit_runner', 'qiskit.ibmq.sampler']

backends = ["qiskit.aer", "qiskit.basicaer", 'qiskit.ibmq']
feature_maps = ["zz", "py", "px", "pz"]
rango = len(feature_maps)

combinaciones = [[backend + "_" + feature_map for feature_map in feature_maps] for backend in backends]

#backends = ["qiskit.aer"]
X, y = get_data()
X_train, X_test, y_train, y_test = split_train_test_data(X=X, y=y)
dispatch_ids = [ct.dispatch(parametros_workflow)(X_train, X_test, y_train, y_test, entrada, rango) for entrada in backends]
#parametros_workflow().draw()
for dispatch_id in dispatch_ids:
    result = ct.get_result(dispatch_id = dispatch_id, wait = True)
    results.append(result)

aciertos = []

for resultado in results:
    for promedio in resultado.result:
        aciertos.append(promedio)

aciertos = np.array(aciertos)
maximo = aciertos.argmax()

mejor_resultado = combinaciones[maximo]
