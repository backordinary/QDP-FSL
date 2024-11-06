# https://github.com/shaqiyu/QC_HEP_Hcc/blob/07c8d8adae4a040596bef9607da5522011bf996c/QHEP/bin/param.py
import ROOT
import numpy as np
import logging
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import SpectralClustering

from sklearn.cluster import SpectralClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import normalized_mutual_info_score, roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score

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
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.colors import Normalize

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

rng = np.random.RandomState(0)

#C_range = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
#gamma_range = [10.0**i for i in range(-8,2)]
C_range = [0.1, 1, 10, 100, 1000, 100000]
gamma_range = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

cv=10

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

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def classification(opt):

    config = Utilities.readConfig("config_qqyy.json")
    
    
    # use -1 for all the events
    nEvent = opt['events']
    sEvents = 0.5
    algorithm1 = "QSVM"
    algorithm2 = "SVM"
    
    algorithm = ["QSVM", "SVM"]
    
    (fpr_SVM, tpr_SVM, fpr_QSVM, tpr_QSVM) = ([] for i in range(4))
    aruc_SVM = None
    aruc_QSVM = None


    aruc_SVMs = []
    aruc_QSVMs= []   
 
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

    n = 100
    mse_list = []
    seed_list = np.arange(0, n)
    
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
           ) = Utilities.preparingData("config_qqyy.json", prossEvent=nEvent, removeNegativeWeight=False, fraction=sEvents, seed=rng, dataType="Quantum") 
           
           
           Plotter.plotVars(config, signal_dataset, bkg_dataset, signal_weights, bkg_weights, "")
           logging.info("The data processed for quantum use in %s", timer() - data_Pro_time)
           
           time_in_qsvm = timer()
           seed = 12345
           
           feature_dim = X.shape[1]
           
           print(
               "Number of qubits used is",
               feature_dim,
               "it should be the number of the variables used in the trainning.",
           )
          
           Gentangle = opt['entangle']
           depth = opt['depth']
           inverse = False
 
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
           # shots=1024
           quantum_instance = QuantumInstance(
               backend, shots=1024, seed_simulator=seed, seed_transpiler=seed
           )
           # qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)
           
           qkernel = QuantumKernel(feature_map=feature_map_cus, quantum_instance=quantum_instance)
           
           ## 
           #qsvcPeg = PegasosQSVC(quantum_kernel=qkernel)
         
           param_grid = dict(gamma=gamma_range, C=C_range)
      
           estim = QSVC(quantum_kernel=qkernel, probability=True)
           #qsvc = GridSearchCV(estimator=estim, param_grid=param_grid, cv=cv, refit=True, verbose=0, scoring='roc_auc')
           qsvc = GridSearchCV(estimator=estim, param_grid=param_grid, refit=True, verbose=0, scoring='roc_auc')


           #clfqPeg = qsvcPeg.fit(X_train, y_train)           
           clfq = qsvc.fit(X_train, y_train)

           print("best_params: %s" % clfq.best_params_)
           print("best_estmator: %s" % clfq.best_estimator_)
           
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
           
           scores = clfq.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))

           fig, ax = plt.subplots()
           im, cbar = Plotter.heatmap(scores, C_range, gamma_range, ax=ax,
                              cmap='RdYlGn_r', cbarlabel="AUC")

           texts = Plotter.annotate_heatmap(im, valfmt="{x:.2f}")
           fig.tight_layout()
           plt.savefig("plots/qqyy/Regularisation_QSVM_c_gamma_{}.pdf".format(nEvent),  bbox_inches = 'tight',
                          pad_inches = 0)

           
           score = clfq.predict_proba(X_test)[:,1]
           print("best_params: %s" % clfq.best_params_)

           means = clfq.cv_results_['mean_test_score']
           stds = clfq.cv_results_['std_test_score']

           with open("plots/qqyy/Regularisation_QSVM_c_gamma_{}_{}.txt".format(cv, nEvent), "w") as f:
               for mean, std, params in zip(means, stds, clfq.cv_results_['params']):
                   print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=f)
                   #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


           training_accuracy_qsvm = clfq.score(X_train, y_train)
           testing_accuracy_qsvm = clfq.score(X_test, y_test)
           
           print(f"\tTraining accuracy QSVM = {training_accuracy_qsvm}")
           print(f"\tTesting accuracy QSVM = {testing_accuracy_qsvm}")
        
    
           y_score_qsvc = clfq.decision_function(X_test)
    
           fpr_QSVM, tpr_QSVM, threshold_QSVM = sklearn.metrics.roc_curve(y_test, y_score_qsvc)
           aruc_QSVM = sklearn.metrics.auc(fpr_QSVM, tpr_QSVM)
           aruc_QSVMs.append(aruc_QSVM)
           print(aruc_QSVMs)

 
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
           ) = Utilities.preparingData("config_qqyy.json", prossEvent=nEvent, removeNegativeWeight=False, fraction=sEvents, seed=rng, dataType="Classical")  #Utilities.preparingData("config_qqyy_old.json", nEvent, False, sEvents, 10, "Classical")
           
    
           #Plotter.plotVars(config, signal_dataset, bkg_dataset, signal_weights, bkg_weights, "Classical")
           logging.info("The data processed for classical use in %s", timer() - data_Pro_time)

           kernel = ['rbf']
           param_grid = dict(kernel=kernel, gamma=gamma_range, C=C_range)
           svc = svm.SVC(kernel='precomputed', probability=True)
           clf = GridSearchCV(estimator=svc, param_grid=param_grid, cv=cv, refit=True, verbose=0, scoring='roc_auc')

           clf.fit(X, y)

           print(
               "The best parameters are %s with a score of %0.2f"
               % (clf.best_params_, clf.best_score_)
           )
           
           scores = clf.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))

           fig, ax = plt.subplots()           
           im, cbar = Plotter.heatmap(scores, C_range, gamma_range, ax=ax,
                              cmap='RdYlGn_r', cbarlabel="AUC")
                              #cmap="YlGn", cbarlabel="AUC")
           texts = Plotter.annotate_heatmap(im, valfmt="{x:.2f}")
           fig.tight_layout()
           plt.savefig("plots/qqyy/Regularisation_SVM_c_gamma_{}.pdf".format(nEvent),  bbox_inches = 'tight',
                          pad_inches = 0)
          
           training_accuracy_svm = clf.score(X_train, y_train)
           testing_accuracy_svm = clf.score(X_test, y_test)

           # new
           score = clf.predict_proba(X_test)[:,1]          
           print("best_params: %s" % clf.best_params_)
  
           means = clf.cv_results_['mean_test_score']
           stds = clf.cv_results_['std_test_score']

           with open("plots/qqyy/Regularisation_SVM_c_gamma_{}.txt".format(nEvent), "w") as f:
               for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                   print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=f)

           #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
           #    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

           predictions = [round(value) for value in score]

           roc_auc =  roc_auc_score(y_test, score)
           acc =  accuracy_score(y_test, predictions)

           print("AUC: %s" % roc_auc)
           print("Accuracy: %s" % acc)


           logging.info(
               "Classical kernel creation and plotting after fitting the data is %s s",
               timer() - time_in_svm,
           )
           
           print(f"\tTraining accuracy SVM = {training_accuracy_svm}")
           print(f"\tTesting accuracy SVM = {testing_accuracy_svm}")
           
           y_score = clf.decision_function(X_test)
           fpr_SVM, tpr_SVM, threshold_SVM = sklearn.metrics.roc_curve(y_test, y_score)
           aruc_SVM = sklearn.metrics.auc(fpr_SVM, tpr_SVM)
    
           print("SVM AUC:", aruc_SVM)
    
    Plotter.plotROCcurve(tpr_QSVM, fpr_QSVM, aruc_QSVM, tpr_SVM, fpr_SVM, aruc_SVM, "ROC_evn%s_%s_param" %(nEvent, Gentangle))


def main():
    global opt
    opt = LoadOptions()
    classification(opt)

if __name__ == '__main__':
    main()
