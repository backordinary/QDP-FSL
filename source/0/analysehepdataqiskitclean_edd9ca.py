# https://github.com/KoljaFrahm/QiskitPerformance/blob/76c7d7e2ae9a4ea7125578a84328658bc6f9ce22/AnalyseHEPdataQiskitClean.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse.sputils import getdata
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches

from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import AerPauliExpectation
from qiskit.circuit.library import TwoLocal, ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
#from IPython.display import clear_output

# callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    #print("callback_graph called")
    #clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.figure()
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.savefig("callback_graph_all.png")
    plt.close()

def onehot(data):
    """
    returns the the data vector in one hot enconding. 
    @data :: Numpy array with data of type int.
    """
    return np.eye(int(data.max()+1))[data]

def index(onehotdata):
    """
    returns the the data vector in index enconding. 
    @onehotdata :: Numpy array with data of type int on onehot encoding.
    """
    return np.argmax(onehotdata, axis=1)

def splitData(x_data, y_data, ratio_train, ratio_test, ratio_val):
    # Produces test split.
    x_remaining, x_test, y_remaining, y_test = train_test_split(
        x_data, y_data, test_size=ratio_test)

    # Adjusts val ratio, w.r.t. remaining dataset.
    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining

    # Produces train and val splits.
    x_train, x_val, y_train, y_val = train_test_split(
        x_remaining, y_remaining, test_size=ratio_val_adjusted)
    
    return x_train, x_test, x_val, y_train, y_test, y_val

def getData():
    data_bkg = np.load("QML-HEP-data/x_data_bkg_8features.npy")
    data_sig = np.load("QML-HEP-data/x_data_sig_8features.npy")

    y_bkg = np.zeros(data_bkg.shape[0],dtype=int)
    y_sig = np.ones(data_sig.shape[0],dtype=int)

    x_data = np.concatenate((data_bkg,data_sig))
    y_data = np.concatenate((y_bkg,y_sig))

    # Defines ratios, w.r.t. whole dataset.
    ratio_train, ratio_test, ratio_val = 0.9, 0.05, 0.05
    x_train, x_test, x_val, y_train, y_test, y_val = splitData(
        x_data, y_data, ratio_train, ratio_test, ratio_val)
    return x_train, x_test, x_val, y_train, y_test, y_val


def plot_roc_get_auc(model,sets,labels):
    aucl = []
    plt.figure()
    for set,label in zip(sets,labels):
        x,y = set
        pred = index(model(x)) #index because the prediction is onehot encoded
        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        plt.plot(fpr, tpr, label=label)
        auc = metrics.roc_auc_score(y, pred)
        aucl.append(auc)
    
    plt.savefig("rocplot.png")
    plt.close()
    return aucl


def VQCTraining(vqc, x_train, y_train, x_test, y_test, epoch, bs):
    # create empty array for callback to store evaluations of the objective function
    plt.figure()
    plt.rcParams["figure.figsize"] = (12, 6)

    #training
    print("fitting starts")
    for epoch in range(epochs):
        batches = gen_batches(x_train.shape[0],bs)
        print(f"Epoch: {epoch + 1}/{epochs}")
        for batch in batches:
            vqc.fit(x_train[batch], onehot(y_train[batch]))
        
        loss = vqc.score(x_test, onehot(y_test))
        print(loss)

    print("fitting finished")

    # return to default figsize
    plt.rcParams["figure.figsize"] = (6, 4)

#declare quantum instance
#simulator_gpu = Aer.get_backend('aer_simulator_statevector')
#simulator_gpu.set_options(device='GPU')
#qi = QuantumInstance(simulator_gpu)
qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector_gpu"))

epochs = 20
nqubits = 8
niter = 20 #how many maximal iterations of the optimizer function
bs = 128  # batch size
#bs = 50  # test batch size

####Define VQC####
feature_map = ZFeatureMap(nqubits, reps=1)
#ansatz = RealAmplitudes(nqubits, reps=1)
ansatz = TwoLocal(nqubits, 'ry', 'cx', 'linear', reps=1)
#optimizer = SPSA(maxiter=niter)
#optimizer = COBYLA(maxiter=niter, disp=True)
optimizer = None

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    loss="cross_entropy",
    optimizer=optimizer,
    warm_start=False,
    quantum_instance=qi,
    callback=callback_graph,
)

print("starting program")
print(vqc.circuit)

x_train, x_test, x_val, y_train, y_test, y_val = getData()


####reduce events for testing the program####
#num_samples = 256
#x_train, x_test, x_val = x_train[:num_samples], x_test[:num_samples], x_val[:num_samples]
#y_train, y_test, y_val = y_train[:num_samples], y_test[:num_samples], y_val[:num_samples]

####VQC#####
objective_func_vals = []
VQCTraining(vqc, x_train, y_train, x_test, y_test, epochs, bs)

####score classifier####
aucs = plot_roc_get_auc(vqc.predict,((x_test, y_test),(x_val,y_val)),("test data","validation data"))
print(aucs)

print("test")
