# https://github.com/shaqiyu/QC_HEP_Hcc/blob/07c8d8adae4a040596bef9607da5522011bf996c/QHEP/bin/run_qsvm_sqlite.py
#!/usr/bin/env python

import os
import traceback
import json
import argparse
import sys
import logging
import time
import datetime
import numpy as np

from utils.load_data import load_events_npy, load_npz_data, save_npz_data

from algorithms.qsvm_sqlite import QSVMDev
from sklearn import svm
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from qiskit import BasicAer
from qiskit import IBMQ, Aer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.components.feature_maps import PauliExpansion, FirstOrderExpansion, SecondOrderExpansion
from qiskit.aqua.utils import split_dataset_to_data_and_labels

from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)

parser = argparse.ArgumentParser()
parser.add_argument('--input', action='store', help='input data', required=True)
parser.add_argument('--training_size', action='store', type=int, default=10000, help='number of events to be selected for training')
parser.add_argument('--test_size', action='store', type=int, default=100000, help='number of events to be selected for test')
parser.add_argument('--nqubits', action='store', type=int, default=5, help='number of qubits')
parser.add_argument('--with_pca', action='store_true', default=False, help='Whether to use PCA')
parser.add_argument('--with_standardscale', action='store_true', default=False, help='Whether to use StandardScale')
parser.add_argument('--shots', action='store', type=int, default=1024, help='number of shots')
parser.add_argument('--feature_map', action='store', default='SecondOrderExpansion', help='feature map method')
parser.add_argument('--entanglement', action='store', default=None, help='entanglement')
parser.add_argument('--feature_map_depth', action='store', type=int, default=3, help='feature map depth')
parser.add_argument('--batch_size', action='store', type=int, default=200, help='batch size')
parser.add_argument('--backend', action='store', default='qasm_simulator', help='backend name')
parser.add_argument('--name', action='store', default='test_qnn', help='problem name')
parser.add_argument('--noise', action='store_true', default=False, help='introducing noise model')
parser.add_argument('--noise_model', action='store', help='noise model', required=False)
parser.add_argument('--cat', action='store', default='tth', help='tth, zerojet, onejet, twojet')
parser.add_argument('--with_error_mitigation', action='store_true', default=False, help='Enable error mitigation')
parser.add_argument('--with_sqlite', action='store_true', default=False, help='Enable local saving')
parser.add_argument('--tokens', action='store', default='', help='IBM API tokens')
parser.add_argument('--degree', action='store', type=int, default=1, help='degree paramter in the featuremap')
parser.add_argument('--jobname', action='store', default='', help='specify the jobname to recover the failed job')
args = parser.parse_args()

##Default Params##
default_params = {
        'problem': {'name': 'vqc', 'random_seed': 10598},
        'loss_function':{'name': 'default'},
        'algorithm': {'name': 'QSVM.Variational', 'max_evals_grouped':2, 'batch_mode': True, 'minibatch_size': 2},
        'backend': {'provider': 'qiskit.BasicAer', 'name': 'qasm_simulator', 'shots': 1024},
        'optimizer': {'name': 'SPSA', 'max_trials': 30, 'save_steps': 1},
        'variational_form': {'name': 'RYRZ', 'depth': 3, 'num_qubits': 5},
        'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'num_qubits': 5, 'entangler_map': None}
}

##Logging##
def get_log_file_name(file, name, args):
    log_dir = os.path.join(os.path.dirname(os.path.realpath(file)), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = "%s_train%s_test%s_qubits%s_pca%s_sscale%s_shots%s_feat%s_entan%s_backend%s.log" % (name,
                                                                                                                    args.training_size,
                                                                                                                    args.test_size,
                                                                                                                    args.nqubits,
                                                                                                                    args.with_pca,
                                                                                                                    args.with_standardscale,
                                                                                                                    args.shots,
                                                                                                                    args.feature_map,
                                                                                                                    args.entanglement,
                                                                                                                    args.backend)
    return os.path.join(log_dir, log_filename)

def config_logging(params):
    log_filename = get_log_file_name(__file__, params['problem']['name'], args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}".format(log_filename)),
            logging.StreamHandler()
        ])
    return log_filename

def load_running_data(filename):
    data = np.load(filename, allow_pickle=True)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test  = data['x_test']
    y_test  = data['y_test']
    return x_train,y_train,x_test,y_test

def main(args):
    ########preparing###############
    params = default_params

    current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    params['problem']['name'] = args.name + "_" + current_time
    params['problem']['random_seed'] = 10598

    #quantum_instance
    params['backend']['shots'] = args.shots
    params['backend']['name']  = args.backend

    #global seed
    aqua_globals.random_seed = params['problem']['random_seed']

    if args.noise:
        IBMQ.load_accounts()
        device = IBMQ.get_backend('ibmq_16_melbourne')
        properties = device.properties()
        from qiskit.providers.aer import noise
        full_noise_model = noise.device.basic_device_noise_model(properties=properties)

        noise_model = noise.utils.remap_noise_model(full_noise_model,
                                                remapping=[i for i in range(1,args.nqubits+1)],
                                                discard_qubits=True)
    else:
       noise_model=None

    basis_gates = None
    red_noise_model = None
    initial_layout = None

    if args.noise_model:
        if args.noise_model.endswith(".npz"):
            nm = np.load(args.noise_model, allow_pickle=True)
            nm_noise = nm['noise_model'].item()
            noise_model = noise.NoiseModel.from_dict(nm_noise)

            coupling_map = nm['coupling_map']
            coupling_map = coupling_map.tolist()
            CMAP = CouplingMap(coupling_map)

            target_qubits = [0,1,2,3,4]
            red_cmap = CMAP.reduce(target_qubits)
            red_noise_model = noise.utils.remap_noise_model(noise_model,
                                                        remapping=target_qubits,
                                                        discard_qubits=True)
            basis_gates = red_noise_model.basis_gates
        else:
            import pickle
            with open(args.noise_model, 'rb') as f:
                 noise_model = pickle.load(f)
            red_noise_model = noise_model
            basis_gates = noise_model.basis_gates

    measurement_error_mitigation_cls = None
    cals_matrix_refresh_period = None

    if args.with_error_mitigation:
       measurement_error_mitigation_cls = CompleteMeasFitter
       cals_matrix_refresh_period = 60

    #feature_map
    params['feature_map']['name'] = args.feature_map
    params['feature_map']['depth'] = args.feature_map_depth
    if args.entanglement == 'linear':
       entangler_map = []
       for i in range(0, args.nqubits - 1, 2):
           entangler_map.append([i, i + 1])
       for i in range(0, args.nqubits - 2, 2):
           entangler_map.append([i + 1, i + 2])
       params['feature_map']['entangler_map'] = entangler_map

    if args.entanglement == 'full':
       params['feature_map']['entangler_map'] = [[i, j] for i in range(args.nqubits) for j in range(i + 1, args.nqubits)]
    if params['feature_map']['name'] == 'SecondOrderExpansion': 
       feature_map = SecondOrderExpansion(feature_dimension=args.nqubits, depth=args.feature_map_depth, entangler_map=params['feature_map']['entangler_map'])
    if params['feature_map']['name'] == 'FirstOrderExpansion':
       feature_map = FirstOrderExpansion(feature_dimension=args.nqubits, depth=args.feature_map_depth)

    if params['feature_map']['name'] == 'FeatureMap7':
       from algorithms.feature_map7 import FeatureMap7
       feature_map = FeatureMap7(num_qubits=args.nqubits, depth=args.feature_map_depth)
    if params['feature_map']['name'] == 'FeatureMap7Dev':
       from algorithms.feature_map_7dev import FeatureMap7Dev
       feature_map = FeatureMap7Dev(num_qubits=args.nqubits, depth=args.feature_map_depth)
    if params['feature_map']['name'] == 'FeatureMaps':
       from algorithms.feature_maps import FeatureMaps
       feature_map = FeatureMaps(num_qubits=args.nqubits, depth=args.feature_map_depth, degree=args.degree)
    if params['feature_map']['name'] == 'FeatureMapsDev':
       from algorithms.feature_maps_dev import FeatureMaps
       feature_map = FeatureMaps(num_qubits=args.nqubits, depth=args.feature_map_depth, degree=args.degree)
    ####################################
    log_filename = config_logging(params)
    logging.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logging.info("+++++++ execution: %s ++++++++++" % params['problem']['name'])
    logging.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logging.info("log file: %s" % log_filename)

 
    logging.info("backend: %s, shots: %s" % (args.backend, str(args.shots)))
    logging.info("feature_map: %s, feature_map_depth: %s, entangler_map:%s, nqubits:%s" % (args.feature_map, str(args.feature_map_depth), params['feature_map']['entangler_map'], str(args.nqubits)))
    logging.info("########qubits: %s," % str(args.nqubits))
    logging.info("########input: %s," % str(args.input))

    if '_running.npz' not in args.input:

        if not args.input.endswith(".npz"):
            sample_Total, training_input, test_input, class_labels = load_events_npy(args.input,
                                                                             args.training_size,
                                                                             args.test_size,
                                                                             args.nqubits,
                                                                             args.with_pca,
                                                                             False,
                                                                             args.with_standardscale,
                                                                             True)
            filename = log_filename.replace('.log', '.npz')
            save_npz_data(filename, training_input, test_input)
        else:
            training_input, test_input = load_npz_data(args.input, args.training_size, args.test_size)

        training_dataset, class_to_label = \
                    split_dataset_to_data_and_labels(training_input)
        test_dataset = split_dataset_to_data_and_labels(test_input, class_to_label)

        x_train, y_train = shuffle(training_dataset[0], training_dataset[1])
        x_test, y_test = shuffle(test_dataset[0], test_dataset[1])
        ##saving the trainning input##
        filename = log_filename.replace('.log', '_running.npz')
        np.savez(filename,x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    else:
        x_train,y_train,x_test,y_test = load_running_data(args.input)

    if len(args.jobname)>1:
       jobname  = args.jobname
    else:
       jobname=log_filename.split("/")[-1].replace('.log', '')

    logging.info("training input size:(%s, %s)" % (len(x_train),len(x_train[0])))
    logging.info("testing input size:(%s, %s)" % (len(x_test),len(x_test[0])))

    #setup qsvm instance 
    qsvm = QSVMDev(feature_map, params['feature_map']['name'], args.batch_size, jobname=jobname, with_sqlite=args.with_sqlite)

    if args.backend.startswith("ibmq_paris")  or args.backend.startswith("ibmq_mumbai"):
        if hasattr(qsvm, '_qr'):
            qr = qsvm._qr
            if args.nqubits == 5:
               initial_layout = {qr[0]: 0, qr[1]: 1, qr[2]: 2, qr[3]: 3, qr[4]: 5}
            elif args.nqubits == 10:
               initial_layout = {qr[0]: 3, qr[1]: 5, qr[2]: 8, qr[3]: 11, qr[4]: 14, qr[5]: 16, qr[6]: 19, qr[7]: 22, qr[8]: 25, qr[9]: 26}
            elif args.nqubits == 15:
               initial_layout = {qr[0]: 3, qr[1]: 5, qr[2]: 8, qr[3]: 11, qr[4]: 14, qr[5]: 16, qr[6]: 19, qr[7]: 22, qr[8]: 25, qr[9]: 24, qr[10]: 23, qr[11]: 21, qr[12]: 18, qr[13]: 15, qr[14]: 12}

    if args.backend.startswith("ibmq_toronto"):
        if hasattr(qsvm, '_qr'):
            qr = qsvm._qr
            if args.nqubits == 5:
               initial_layout = {qr[0]: 0, qr[1]: 1, qr[2]: 4, qr[3]: 7, qr[4]: 10}
            elif args.nqubits == 10:
               initial_layout = {qr[0]: 3, qr[1]: 5, qr[2]: 8, qr[3]: 11, qr[4]: 14, qr[5]: 16, qr[6]: 19, qr[7]: 22, qr[8]: 25, qr[9]: 26}

    if args.backend.startswith("ibmq_johannesburg"):
        if hasattr(qsvm, '_qr'):
            qr = qsvm._qr
            if args.nqubits == 5:
               initial_layout = {qr[0]: 0, qr[1]: 5, qr[2]: 6, qr[3]: 7, qr[4]: 8}
            elif args.nqubits == 10:
               initial_layout = {qr[0]: 0, qr[1]: 5, qr[2]: 6, qr[3]: 7, qr[4]: 8, qr[5]: 9, qr[6]: 14, qr[7]: 13, qr[8]: 12, qr[9]: 11}

    if args.backend.startswith("ibmq_rome"):
        if hasattr(qsvm, '_qr'):
            qr = qsvm._qr
            initial_layout = {qr[0]: 0, qr[1]: 1, qr[2]: 2, qr[3]: 3, qr[4]: 4}

    #setup quantum instance
    if 'ibmq' in args.backend:
       from utils import Qconfig
       try:
           if len(args.tokens)>0:
              APItoken = args.tokens
           else:
              APItoken = Qconfig.APItoken

           IBMQ.enable_account(APItoken, Qconfig.config['url'])
       except:
           pass
       my_provider = IBMQ.get_provider(hub=Qconfig.config['hub'], group=Qconfig.config['group'], project=Qconfig.config['project'])
       #my_provider = IBMQ.get_provider()

       quantum_instance = QuantumInstance(my_provider.get_backend(name=params['backend']['name']),
                                       shots=params['backend']['shots'],
                                       basis_gates=basis_gates,
                                       noise_model=red_noise_model,
                                       seed_simulator=params['problem']['random_seed'],
                                       seed_transpiler=params['problem']['random_seed'],
                                       initial_layout=initial_layout,
                                       wait=30,
                                       optimization_level=1,
                                       measurement_error_mitigation_cls=measurement_error_mitigation_cls,
                                       cals_matrix_refresh_period = cals_matrix_refresh_period)
    else:
       quantum_instance = QuantumInstance(Aer.get_backend(params['backend']['name']),
                                          shots=params['backend']['shots'],
                                          basis_gates=basis_gates,
                                          noise_model=red_noise_model,
                                          initial_layout=initial_layout,
                                          seed_simulator=params['problem']['random_seed'],
                                          seed_transpiler=params['problem']['random_seed'],
                                          optimization_level=1,
                                          measurement_error_mitigation_cls=measurement_error_mitigation_cls,
                                          cals_matrix_refresh_period = cals_matrix_refresh_period)

    train_kernel_matrices = []
    test_kernel_matrices = []

    train_kernel_matrix= qsvm.get_kernel_matrix(quantum_instance, feature_map, x_train, None)
    test_kernel_matrix = qsvm.get_kernel_matrix(quantum_instance, feature_map, x_test, x_train)

    svc = svm.SVC(kernel='precomputed', probability=True, random_state=42)

    tune_params = {
                   'C': [1,2,3,4,5,8,9,10,15,20,30,50,100,200,400,800,1000],
                  }

    #tune_params = {
    #               'C': [2,3,4,5,8,9,10,15,20,30,50,100,200,400,800,1000],
    #              }

    clf = GridSearchCV(estimator=svc, param_grid=tune_params, n_jobs=-1, cv=3, scoring='roc_auc')

    clf.fit(train_kernel_matrix,y_train)

    score = clf.predict_proba(test_kernel_matrix)[:,1]

    logging.info("best_params: %s" % clf.best_params_)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        logging.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    predictions = [round(value) for value in score]

    roc_auc =  roc_auc_score(y_test, score)
    acc =  accuracy_score(y_test, predictions)

    logging.info("AUC: %s" % roc_auc)
    logging.info("Accuracy: %s" % acc)

    result_filename = log_filename.replace('.log', '_auc_result.npz')

    logging.info("path of the result: %s" % result_filename)

    np.savez(result_filename,train_kernel_matrix=train_kernel_matrix,test_kernel_matrix=test_kernel_matrix, y_test=y_test,score=score)

if __name__ == '__main__':
    main(args)
