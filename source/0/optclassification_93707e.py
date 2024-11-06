# https://github.com/shaqiyu/QC_HEP_Hcc/blob/07c8d8adae4a040596bef9607da5522011bf996c/QHEP/bin/Optclassification.py
import ROOT
import numpy as np
import logging
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

from qiskit import Aer, BasicAer
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn import svm
import os
import sys, re
from array import array
from optparse import OptionParser

# this_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, f"{this_dir}/../qiskit_hep_plotting")
from Qhep_Modules import Plotter, Utilities, customised_feature_maps

import logging.config

# from feature_maps_from_Chen import FeatureMaps

from timeit import default_timer as timer

#os.environ["MPLCONFIGDIR"] = "/tmp/"

logging.basicConfig(
    filename="logs/log",
    filemode="w",
    format="%(levelname)s:%(message)s",
    level=logging.DEBUG,
)

data_Pro_time = timer()

logging.info("Start time %s", data_Pro_time)

try:
    logging.info("Loading ROOT module...")
    import ROOT

    ROOT.gSystem.Load("libRooFit")  # not necessary, but neatens up the output
    ROOT.gROOT.SetBatch(True)  # to prevent TCanvases showing up
except ImportError:
    logging.error(
        "Could not import ROOT module. Make sure your root is configured to work with Python."
    )
    sys.exit()



def LoadOptions():
    parser = OptionParser()

    parser.add_option("-e","--Entangle",dest="entanglment", default="",type=str, help='Use from: fullforward, fullbackward')
    parser.add_option("-n","--Events",dest="events_number", default=100 ,type=int, help='Enter the number of events you want to use, 100, 200, 400, ...')
    parser.add_option("-d","--Depth",dest="depth_number", default=1 ,type=int, help='Enter the number of circuit repetition you want to use, 1, 2, 3, ...')


    (options, args) = parser.parse_args()


    return { 'entangle' :   options.entanglment,
             'events' :   options.events_number,
             'depth' :   options.depth_number
             #'inverse' :   options.inverse_circuit
           }

n = 101
seed_list = np.arange(0, n)

def classification(opt):

    config = Utilities.readConfig("config_qqyy_old.json")
    #config = Utilities.readConfig("config_qqyy.json")
    
    
    # use -1 for all the events
    nEvent = opt['events']
    sEvents = 0.5
    algorithm1 = "QSVM"
    algorithm2 = "SVM"
    
    algorithm = ["QSVM", "SVM"]
    
    (fpr_SVM, tpr_SVM, fpr_QSVM, tpr_QSVM) = ([] for i in range(4))
    aruc_SVM = None
    aruc_QSVM = None
    
    # Create empty lists
    (
        X_train,
        X_test,
        y_train,
        y_test,
        XW_train,
        XW_test,
        signal_dataset,
        bkg_dataset,
        signal_weights,
        bkg_weights,
        X,
        y,
    ) = ([] for i in range(12))
    
    train_dataset = {}
    test_dataset = {}

    mse_list_qsvm = []
    mse_list_svm = []

    for seeds in seed_list:    
       for algo in algorithm:
           if algo=="QSVM":
           
              (
                  X_train,
                  X_test,
                  y_train,
                  y_test,
                  XW_train,
                  XW_test,
                  train_dataset,
                  test_dataset,
                  signal_dataset,
                  bkg_dataset,
                  signal_weights,
                  bkg_weights,
                  X,
                  y,
              ) = Utilities.preparingData("config_qqyy_old.json", nEvent, False, sEvents, seeds, "Quantum")
              
              
              Plotter.plotVars(config, signal_dataset, bkg_dataset, signal_weights, bkg_weights, "")
              logging.info("The data processed for quantum use in %s", timer() - data_Pro_time)
              
              time_in_qsvm = timer()
              seed = 12345
              # algorithm_globals.random_seed = seed
              
              
              feature_dim = X.shape[1]
              
              print(
                  "Number of qubits used is",
                  feature_dim,
                  "it should be the number of the variables used in the trainning.",
              )
             
              Gentangle = opt['entangle']
              depth = opt['depth']
              inverse = False #opt['inverse']
 
              feature_map_cus = customised_feature_maps.FeatureMap(
                  num_qubits=feature_dim, depth=depth, degree=1, entanglement=Gentangle, inverse=False
              )


              #feature_map_cus = customised_feature_maps.FeatureMap_2(
              #    num_qubits=4, depth=depth, degree=1, entanglement=Gentangle, inverse=inverse
              #)
              
              print("Custom feature map:\n", feature_map_cus)
              
              #feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement="linear")
              #print("ZZFeature map:\n", feature_map)
              
              # backend = BasicAer.get_backend('qasm_simulator')       # Very slow
              backend = BasicAer.get_backend("statevector_simulator")  # fast
              
              quantum_instance = QuantumInstance(
                  backend, shots=1024, seed_simulator=seed, seed_transpiler=seed
              )
              # qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)
              
              qkernel = QuantumKernel(feature_map=feature_map_cus, quantum_instance=quantum_instance)
              
              ## 
              qsvc = QSVC(quantum_kernel=qkernel, probability=True)
              #qsvc = PegasosQSVC(quantum_kernel=qkernel)
              
              clfq = qsvc.fit(X_train, y_train)
              
              qsvm_kernel_matrix_train = qkernel.evaluate(x_vec=X_train)
              qsvm_kernel_matrix_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)
              
              QMatrix1 = plt.figure(figsize=(5, 5))
              plt.figure(1)
              plt.imshow(
                  np.asmatrix(qsvm_kernel_matrix_train),
                  interpolation="nearest",
                  origin="upper",
                  cmap="copper_r",
              )
              plt.title("QSVC clustering kernel matrix (training)")
              plt.savefig(
                  "plots/qqyy/Kernel/Qsvm_clustering_kernel_matrinx_train_{}events.pdf".format(nEvent)
              )
              
              QMatrix2 = plt.figure(figsize=(5, 5))
              plt.figure(2)
              plt.imshow(
                  np.asmatrix(qsvm_kernel_matrix_test),
                  interpolation="nearest",
                  origin="upper",
                  cmap="copper_r",
              )
              plt.title("QSVC clustering kernel matrix (testing)")
              plt.savefig(
                  "plots/qqyy/Kernel/Qsvm_clustering_kernel_matrinx_evn{}_{}.pdf".format(nEvent, Gentangle)
              )
              
              logging.info(
                  "Quantum kernel creation and plotting after fitting the data is %s s",
                  timer() - time_in_qsvm,
              )
              
              # y=1 is the signal label and y=-1 is the background label
              #X_signal = X[y == 1]
              #X_background = X[y == -1]
              
              #qsvc_scores_signal = clfq.decision_function(X_signal)
              #qsvc_scores_background = clfq.decision_function(X_background)
              
              training_accuracy_qsvm = clfq.score(X_train, y_train)
              testing_accuracy_qsvm = clfq.score(X_test, y_test)
              
              print(f"\tTraining accuracy QSVM = {training_accuracy_qsvm}")
              print(f"\tTesting accuracy QSVM = {testing_accuracy_qsvm}")
           
       
              y_score_qsvc = clfq.decision_function(X_test)
       
              fpr_QSVM, tpr_QSVM, threshold_QSVM = sklearn.metrics.roc_curve(y_test, y_score_qsvc)
              aruc_QSVM = sklearn.metrics.auc(fpr_QSVM, tpr_QSVM)
           
              mse_list_qsvm.append(aruc_QSVM)
              print("QSVM AUC:", aruc_QSVM)
              
              time_in_svm = timer()
           
           elif algo=="SVM":
              print("SVM algorithm",algo)
              #--------------
              #Classical SVM
              #--------------
           
              (
                  X_train,
                  X_test,
                  y_train,
                  y_test,
                  XW_train,
                  XW_test,
                  train_dataset,
                  test_dataset,
                  signal_dataset,
                  bkg_dataset,
                  signal_weights,
                  bkg_weights,
                  X,
                  y,
              ) = Utilities.preparingData("config_qqyy_old.json", nEvent, False, sEvents, seeds, "Classical")
              
       
       
              #Plotter.plotVars(config, signal_dataset, bkg_dataset, signal_weights, bkg_weights, "Classical")
              logging.info("The data processed for classical use in %s", timer() - data_Pro_time)
       
              clf = svm.SVC(kernel="rbf", probability=True)  # Linear, rbf Kernel
              
              clf.fit(X_train, y_train)
              
              y_train_pre = clf.predict(X_train)
              y_test_pre = clf.predict(X_test)
             
              training_accuracy_svm = clf.score(X_train, y_train)
              testing_accuracy_svm = clf.score(X_test, y_test)
              
              logging.info(
                  "Classical kernel creation and plotting after fitting the data is %s s",
                  timer() - time_in_svm,
              )
              
              print(f"\tTraining accuracy SVM = {training_accuracy_svm}")
              print(f"\tTesting accuracy SVM = {testing_accuracy_svm}")
              
              y_score = clf.decision_function(X_test)
              fpr_SVM, tpr_SVM, threshold_SVM = sklearn.metrics.roc_curve(y_test, y_score)
              aruc_SVM = sklearn.metrics.auc(fpr_SVM, tpr_SVM)
       
              mse_list_svm.append(aruc_SVM)

              print("SVM AUC:", aruc_SVM)
       
       Plotter.plotROCcurve(tpr_QSVM, fpr_QSVM, aruc_QSVM, tpr_SVM, fpr_SVM, aruc_SVM, "ROC_evn%s_%s" %(nEvent, Gentangle))


    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    plt.plot(seed_list, mse_list_qsvm, "b", label="QSVM")
    plt.plot(seed_list, mse_list_svm, "r--", label="SVM")
    plt.title("")
    plt.xlabel('Random State')
    plt.ylabel('AUC')
    plt.grid()
    ax.tick_params(axis="x", direction="in", length=6)
    ax.tick_params(axis="y", direction="in", length=6)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('plots/qqyy/mse_vs_random_state_%s-%s.pdf' %(min(seed_list), max(seed_list)))

    for i in range(len(seed_list)):
       print("Seed:", seed_list[i],"QAUC:",mse_list_qsvm[i],"AUC:",mse_list_svm[i])

# Old python code for plotting
#plt.figure(figsize=(5, 5))
#plt.figure(5)
#ax = plt.gca()
#plt.plot([0, 1], [0, 1], "r--")
#plt.plot(tpr_qsvc, 1 - fpr_qsvc, "b", label="QSVM (AUC = %0.3f)" % (qsvm_aruc))
#plt.plot(tpr, 1 - fpr, "k", label="SVM (AUC = %0.3f)" % (svm_aruc))
#plt.xlabel("Signal efficiency")
#plt.ylabel("Background rejection")
## plt.title('ROC curve')
#ax.set_xlabel("Signal efficiency", loc="right")
#ax.tick_params(axis="x", direction="in", length=6)
#ax.tick_params(axis="y", direction="in", length=6)
#ax.xaxis.set_ticks_position("both")
#ax.yaxis.set_ticks_position("both")
#plt.legend(loc="best")
#plt.savefig("./plots/qqyy/ROCs/ROC_{}events.pdf".format(nEvent))
#
## Response and overtraining
## QSVM
#signal_Res_train_QSVC = clfq.predict_proba(X_train)[:, 1]
#background_Res_train_QSVC = clfq.predict_proba(X_train)[:, 0]
#
#signal_Res_test_QSVC = clfq.predict_proba(X_test)[:, 1]
#background_Res_test_QSVC = clfq.predict_proba(X_test)[:, 0]
#
## SVM
#signal_Res_train_SVC = clf.predict_proba(X_train)[:, 1]
#background_Res_train_SVC = clf.predict_proba(X_train)[:, 0]
#
#signal_Res_test_SVC = clf.predict_proba(X_test)[:, 1]
#background_Res_test_SVC = clf.predict_proba(X_test)[:, 0]
#
#Plotter.plotOvertraining(
#    signal_Res_train_QSVC,
#    background_Res_train_QSVC,
#    signal_Res_test_QSVC,
#    background_Res_test_QSVC,
#    XW_train,
#    XW_test,
#    "QSVM_{}events".format(nEvent),
#)
#Plotter.plotOvertraining(
#    signal_Res_train_SVC,
#    background_Res_train_SVC,
#    signal_Res_test_SVC,
#    background_Res_test_SVC,
#    XW_train,
#    XW_test,
#    "SVM_{}events".format(nEvent),
#)
#
#logging.info("Total execution time %s min", (timer() - data_Pro_time) / 60.0)

def main():
    global opt
    opt = LoadOptions()
    classification(opt)

if __name__ == '__main__':
    main()
