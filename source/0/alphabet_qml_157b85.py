# https://github.com/wweronika/qml_language_guessing/blob/f23e650391ae1c640f356e598977f03db6c3405c/alphabet_QML.py
import numpy as np

from polyadicqml import Classifier
from polyadicqml.qiskit.utility import Backends
from polyadicqml.qiskit import qkCircuitML
from polyadicqml.manyq import mqCircuitML
from qiskit import Aer

from dataLoader import loadData

X_train, X_test, y_train, y_test = loadData("data.csv","languages.csv")


def languageCircuit(bdr, x, p):
    bdr.allin(x[[0,1]])
    bdr.cz(0,1).allin(p[[0,1]])
    #from 2 to 25 inclusive in intervals of 2
    for i in range(2, 26, 2):
        bdr.cz(0,1).allin(x[[i, i+1]])
        bdr.cz(0,1).allin(p[[i, i+1]])
    return bdr


nbqbits = 2
nbparams = 26

qc = mqCircuitML(make_circuit=languageCircuit, nbqbits=nbqbits, nbparams=nbparams)

bitstr = ['00', '01', '10', '11']

model = Classifier(qc, bitstr).fit(X_train, y_train, method="BFGS")

backend = Aer.get_backend('qasm_simulator')

qc = qkCircuitML(
    make_circuit=languageCircuit,
    nbqbits=nbqbits, nbparams=nbparams,
    backend=backend
)

model.set_circuit(qc)
model.nbshots = 330
model.job_size = 30

pred_train = model(X_train)
pred_test = model(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

def print_results(target, pred, name="target"):
    print('\n' + 30*'#',
        "Confusion matrix on {}:".format(name), confusion_matrix(target, pred),
        "Accuracy : " + str(accuracy_score(target, pred)),
        sep='\n')

print_results(y_train, pred_train, name="train")
print_results(y_test, pred_test, name="test")



