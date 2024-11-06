# https://github.com/shaqiyu/QC_HEP_Hcc/blob/efbbed4b8386059b0f13988494e5e030fbf58ccd/QHEP/bin/classification.py
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
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import normalized_mutual_info_score, roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import plot_roc_curve
from qiskit import Aer, BasicAer
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import os
import sys, re
from array import array
from optparse import OptionParser

# this_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, f"{this_dir}/../qiskit_hep_plotting")
from Qhep_Modules_0820 import Plotter, Utilities, customised_feature_maps

import logging.config
#==============
#  IBM
#==============
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, transpile, Aer, assemble
from qiskit.tools.monitor import job_monitor
import warnings

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

rng = 45
#rng = np.random.RandomState(0)

def LoadOptions():
    parser = OptionParser()

    parser.add_option("-e","--Entangle",dest="entanglment", default="Full",type=str, help='Use from: fullforward, fullbackward')
    parser.add_option("-n","--Events",dest="events_number", default=100 ,type=int, help='Enter the number of events you want to use, 100, 200, 400, ...')
    parser.add_option("-d","--Depth",dest="depth_number", default=1 ,type=int, help='Enter the number of circuit repetition you want to use, 1, 2, 3, ...')
    parser.add_option("-b","--backend",dest="backend", default="ibmq" ,type=str, help='Use ibm quantum computor')
    parser.add_option("-c","--Cnumb",dest="C_number", default=1 ,type=float, help='Use C in SVM')
    parser.add_option("-g","--gamma_numb",dest="gamma_number", default=0.1 ,type=float, help='Use gamma in SVM')

    (options, args) = parser.parse_args()


    return { 'entangle' :   options.entanglment,
             'events'   :   options.events_number,
             'depth'    :   options.depth_number,
             'backend'  :   options.backend,
             'cnumb'    :   options.C_number,           
             'gamma'    :   options.gamma_number           
           }

def classification(opt):

    varSetting = "config_vvH_9.json"

    config = Utilities.readConfig("config_vvH_9.json")
    
    
    # use -1 for all the events
    nEvent = opt['events']
    sEvents = 1
    cnumb = opt['cnumb']
    gamma_numb = opt['gamma']
    
    #algorithm = ["QSVM","SVM"]
    #algorithm = ["QSVM"]
    algorithm = ["SVM"]
    
    (fpr_SVM, tpr_SVM, fpr_QSVM, tpr_QSVM) = ([] for i in range(4))

    mean_fpr = np.linspace(0, 1, 100000)
    cv = StratifiedKFold(n_splits=3)

    tprsq = []
    tprsq_ibm = []

    tprs  = []

    aucs  = []
    aucsq = []
    aucsq_ibm = []

    aruc_SVMs = []
    aruc_QSVMs= []
    aruc_QSVMs_ibm= []

    std_auc = 0
    std_qauc_ibm = 0
    std_qauc = 0

    aruc_SVM = None
    aruc_QSVM = None
    aruc_QSVM_ibm = None

    # Create empty lists
    (
        X_train,
        X_test,
        y_train,
        y_test,
        signal_dataset,
        bkg_dataset,
        X,
        y,
    ) = ([] for i in range(8))
    
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
                signal_dataset,
                bkg_dataset,
                X,
                y,
            ) = Utilities.preparingData(varSetting, prossEvent=nEvent, fraction=sEvents, seed=rng, dataType="Quantum")
           
           
            #print("signal_dataset:")
            #print(signal_dataset)
            #print("bkg_dataset:")
            #print(bkg_dataset)
            #Plotter.plotVars(config, signal_dataset, bkg_dataset, "")
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

            #================================================#
            # Wheather to use real quantum computer or not   #
            #================================================#
           
            backend = None
            feature_map_cus = None
            qkernel = None

            if "both" in opt['backend']:
                warnings.filterwarnings('ignore')
                IBMQ.save_account("75685791c48132479d6c609db91a6f636051da392dba2f4dde1190c8b1f222aead93df9c00bb30fb1df78a7789d59b52844fefba66d0d3681683dcad95080e3f", overwrite=True)
                provider = IBMQ.load_account()

                provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

                feature_map_cus = customised_feature_maps.FeatureMap(
                    num_qubits=feature_dim, depth=depth, degree=1, entanglement=Gentangle, inverse=False
                )

                print("Custom feature map:\n", feature_map_cus)

                backend_ibm = provider.get_backend('ibmq_santiago') #ibmq_lima

                mapped_circuit = transpile(feature_map_cus, backend=backend_ibm)
                quantum_instance_ibm = QuantumInstance(backend_ibm, shots=1024) 
                qkernel_ibm = QuantumKernel(feature_map=mapped_circuit, quantum_instance=quantum_instance_ibm) 

                backend = BasicAer.get_backend("statevector_simulator")  # fast

                quantum_instance = QuantumInstance(
                    backend, shots=1024, seed_simulator=seed, seed_transpiler=seed
                )
                qkernel = QuantumKernel(feature_map=feature_map_cus, quantum_instance=quantum_instance)

                qsvc = QSVC(quantum_kernel=qkernel, probability=True, C=30)
                qsvc_ibm = QSVC(quantum_kernel=qkernel_ibm, probability=True, C=30)

                for i, (train, test) in enumerate(cv.split(X, y)):
                    clfq_ibm = qsvc_ibm.fit(X[train], y[train])
                    clfq     = qsvc.fit(X[train], y[train])
                    scores_ibm = clfq_ibm.decision_function(X[test])
                    scores     = clfq.decision_function(X[test])

                    fpr_ibm, tpr_ibm, threshold_ = sklearn.metrics.roc_curve(y[test], scores_ibm)
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(y[test], scores)

                    roc_auc_ibm = sklearn.metrics.auc(fpr_ibm, tpr_ibm)
                    roc_auc = sklearn.metrics.auc(fpr, tpr)


                    print("ibm ROC fold %d (AUC = %0.3f)" %(i+1, roc_auc_ibm))
                    print("ROC fold %d (AUC = %0.3f)" %(i+1, roc_auc))


                    interp_tpr = np.interp(mean_fpr, fpr, tpr)

                    interp_tpr[0] = 0.0
                    tprsq_ibm.append(interp_tpr)
                    aruc_QSVMs_ibm.append(roc_auc_ibm)
                    tprsq_ibm.append(interp_tpr)
                    aruc_QSVMs_ibm.append(roc_auc)


                tpr_QSVM_ibm = np.mean(tprsq_ibm, axis=0)
                fpr_QSVM_ibm = mean_fpr_ibm
                tpr_QSVM = np.mean(tprsq, axis=0)
                fpr_QSVM = mean_fpr


                mean_qauc_ibm = sklearn.metrics.auc(fpr_QSVM_ibm, tpr_QSVM_ibm)
                mean_qauc = sklearn.metrics.auc(fpr_QSVM, tpr_QSVM)

                aruc_QSVM_ibm = np.mean(aruc_QSVMs_ibm)
                std_qauc_ibm = np.std(aruc_QSVMs_imb)

                aruc_QSVM = np.mean(aruc_QSVMs)
                std_qauc = np.std(aruc_QSVMs)


                print("QSVM AUC IBM:", aruc_QSVM_ibm)
                print("QSVM AUC Mean IBM: %s" % mean_qauc_ibm)
                print("Standard deviation for quantum IBM: %s" % np.std(aruc_QSVMs_ibm))

                print("QSVM AUC:", aruc_QSVM)
                print("QSVM AUC Mean: %s" % mean_qauc)
                print("Standard deviation for quantum: %s" % np.std(aruc_QSVMs))


            elif 'sim' in opt['backend']:
                backend = BasicAer.get_backend("statevector_simulator")  

                feature_map_cus = customised_feature_maps.FeatureMap(num_qubits=feature_dim, depth=depth, degree=1, inverse=False)

                print("Custom feature map:\n", feature_map_cus)
                #print("C_number:\n", cnumb)
                #print("gamma_number:\n", gamma_numb)

                quantum_instance = QuantumInstance(
                    backend, shots=1024, seed_simulator=seed, seed_transpiler=seed
                )

                qkernel = QuantumKernel(feature_map=feature_map_cus, quantum_instance=quantum_instance)

 
                #qsvc = QSVC(quantum_kernel=qkernel, probability=False, C=cnumb, verbose=False)

                #tuned_parameters = [{'C': [45]}]
                #tuned_parameters = [{'C': [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,50]}]
                tuned_parameters = [{'C': [1]}]
                #tuned_parameters = [{'C': [50,55,60,65,70,75,80]}]
                #tuned_parameters = [{'C': [85,90,95,100,200,300,400,500,600,1000]}]
                scores = ['roc_auc']
                for score in scores:
                    print("# Tuning hyper-parameters for %s" % score)
                    print()

                 # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
                    clfq = GridSearchCV(QSVC(quantum_kernel=qkernel, probability=True, verbose=False),tuned_parameters, cv=3, scoring='%s' % score)
                # 用训练集训练这个学习器 clf
                    clfq.fit(X_train, y_train)
                    #clfq.fit(X_test, y_test)

                    print("Best parameters set found on development set:")
                    print()

                # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
                    print(clfq.best_params_)

                    print()
                    print("Grid scores on development set:")
                    print()
                    means = clfq.cv_results_['mean_test_score']
                    stds = clfq.cv_results_['std_test_score']

                # 看一下具体的参数间不同数值的组合后得到的分数是多少
                    for mean, std, params in zip(means, stds, clfq.cv_results_['params']):
                        print("%0.3f (+/-%0.03f) for %r"
                          % (mean, std * 2, params))

                    print()

                    print("Detailed classification report:")
                    print()
                    print("The model is trained on the full development set.")
                    print("The scores are computed on the full evaluation set.")
                    print()
                    y_true, y_pred = y_test, clfq.predict(X_test)

                # 打印在测试集上的预测结果与真实值的分数
                    print(classification_report(y_true, y_pred))

                    print()

                    y_score = clfq.decision_function(X_test)
                    fpr_qsvm, tpr_qsvm, threshold = sklearn.metrics.roc_curve(y_test, y_score)
                    print(fpr_qsvm)
                    print(tpr_qsvm)
                    print(threshold)
                    aruc_qsvm = sklearn.metrics.auc(fpr_qsvm, tpr_qsvm)
                    std_qsvm = stds[0]


                print("**********************************************************") 
                print("*******      Results for the simulator only        *******")
                print("**********************************************************")
              #  print("QSVM AUC:", aruc_QSVM)
              #  print("QSVM AUC Mean: %s" % mean_qauc)
              #  print("Standard deviation for quantum: %s" % np.std(aruc_QSVMs))
                print("**********************************************************")


            elif 'noQSVM' in opt['backend']:  
            
                print("**********************************************************") 
                print("*******             NO QSVM results           ************")
                print("**********************************************************")
                print("C_number:\n", cnumb)
                print("gamma_number:\n", gamma_numb)



           #=================================================#
           #			Classical SVM                 #
           #=================================================#
            time_in_svm = timer()
        elif algo=="SVM":
            print("SVM algorithm",algo)
                   
            (
               X_train,
               X_test,
               y_train,
               y_test,
               signal_dataset,
               bkg_dataset,
               X,
               y,
            ) = Utilities.preparingData(varSetting, prossEvent=nEvent, fraction=sEvents, seed=rng, dataType="Classical")
            print(rng) 
            logging.info("The data processed for classical use in %s", timer() - data_Pro_time)
   
            svc = svm.SVC(kernel='rbf', probability=False, C=cnumb, gamma=gamma_numb, verbose=True)
            #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            #         'C': [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,50,55,60,65,70,75,80,85,90,95,100]}]
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1],
                     'C': [1]}]
            scores = ['roc_auc']
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print()

                 # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
                clf = GridSearchCV(SVC(), tuned_parameters, cv=3,scoring='%s' % score, n_jobs=-1,refit=True)
                # 用训练集训练这个学习器 clf
                #clf.fit(X_test, y_test)
                clf.fit(X_train, y_train)

                print("Best parameters set found on development set:")
                print()

                # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
                print(clf.best_params_)
                print(clf.best_score_)

                print()
                print("Grid scores on development set:")
                print()
                means_svm = clf.cv_results_['mean_test_score']
                stds_svm = clf.cv_results_['std_test_score']

                # 看一下具体的参数间不同数值的组合后得到的分数是多少
                for mean_svm, std_svm, params in zip(means_svm, stds_svm, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean_svm, std_svm * 2, params))

                print()

                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
                y_true, y_pred = y_test, clf.predict(X_test)

                # 打印在测试集上的预测结果与真实值的分数
                print(classification_report(y_true, y_pred))

                print()
                y_score = clf.decision_function(X_test)
                fpr_svm,tpr_svm,threshold = sklearn.metrics.roc_curve(y_test, y_score)
                #print(fpr_svm)
                #print(tpr_svm)
                #print(threshold)
                aruc_svm = sklearn.metrics.auc(fpr_svm,tpr_svm)
                std_svm = stds_svm[0]
#            for i, (train, test) in enumerate(cv.split(X, y)):
#                clf = svc.fit(X[train], y[train])
#                scores = clf.decision_function(X[test])
#                fpr, tpr, threshold = sklearn.metrics.roc_curve(y[test], scores)
#                roc_auc = sklearn.metrics.auc(fpr, tpr)
#
#                print("ROC fold %d (AUC = %0.3f)" %(i+1, roc_auc))
#
#                interp_tpr = np.interp(mean_fpr, fpr, tpr)
#
#                interp_tpr[0] = 0.0
#                tprs.append(interp_tpr)
#                aruc_SVMs.append(roc_auc)
#
#
#            tpr_SVM = np.mean(tprs, axis=0)
#            fpr_SVM = mean_fpr
#
#
#            mean_auc = sklearn.metrics.auc(fpr_SVM, tpr_SVM)
#
#            aruc_SVM = np.mean(aruc_SVMs)
#            std_auc = np.std(aruc_SVMs)
#
#            print("                                                          ")
#            print("**********************************************************")
#            print("*******      Results for the clasical SVM          *******")
#            print("**********************************************************")
#            print("SVM AUC mean:", aruc_SVM)
#            print("SVM AUC check: %s" % mean_auc)
#            print("Standard deviation: %s" % np.std(aruc_SVMs))
#            print("**********************************************************")
#
#            logging.info(
#                "Classical kernel creation and plotting after fitting the data is %s s",
#                timer() - time_in_svm,
#            )
           
    #if "both" in opt['backend']: 
    #   plotROC_IBM(tpr_QSVM, fpr_QSVM, aruc_QSVM, std_qauc, tpr_QSVM_ibm, fpr_QSVM_ibm, aruc_QSVM_ibm, std_qauc_ibm, tpr_SVM, fpr_SVM, aruc_SVM, std_auc, "ROC_evn%s_%s_class_2" %(nEvent, Gentangle)) 
   # else:
    if "sim" in opt['backend']: 
              Plotter.plotROCcurve(tpr_qsvm, fpr_qsvm, aruc_qsvm, std_qsvm, tpr_svm, fpr_svm, aruc_svm, std_svm, "ROC_C%s_N%s_%s" %(cnumb, nEvent, Gentangle))


def main():
    global opt
    opt = LoadOptions()
    classification(opt)

if __name__ == '__main__':
    main()
