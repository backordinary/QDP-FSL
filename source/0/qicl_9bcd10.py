# https://github.com/ezeeEric/qicl/blob/1d87f6a077f382bce964e0f88bd331d29e7e4981/aux/qicl.py
#!/usr/bin/env python
# coding: utf-8

# This is a test file for the VQC implementation of a HEP classification problem.
# @author: Eric Drechsler (dr.eric.drechsler@gmail.com)

# Copied from qiskit-community-tutorials

import pandas
import pickle
import time
time0=time.time()

from datasets import *
from qiskit import BasicAer
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.input import ClassificationInput
import numpy as np

from plotter import plotTruth,plotVars

from argparse import ArgumentParser

argParser = ArgumentParser(add_help=False)
argParser.add_argument( '-t', '--steerTestRun', action="store_true")
argParser.add_argument( '-od', '--steerOutDir', help='Output Directory', default=".")
argParser.add_argument( '-nevt', '--numberEvents', help='Number of events', default=10)
argParser.add_argument( '-sh',   '--numberShots', help='Number of shots', default=1024)
argParser.add_argument( '-mt',   '--maxTrials', help='Max trials SPSA', default=20)
argParser.add_argument( '-ss',   '--saveSteps', help='SPSA save steps', default=5)
argParser.add_argument( '-vfd',  '--varFormDepth', help='variational form depth', default=2)
argParser.add_argument( '-fmd',  '--featMapDepth', help='Feature Map depth', default=1)
#default
#argParser.add_argument( '-spsa',  '--spsaoptim', help='Use SPSA optimiser', action='store_true')
argParser.add_argument( '-mop',  '--steerManualOptim', help='Use manual optimisation strategy', action='store_true')
args = argParser.parse_args()

in_df = pandas.read_pickle("MixData_PD.pkl")

nevt = int(args.numberEvents)
var1 = 'lep1_pt'
var2 = 'lep2_pt'
var3 = 'reco_zv_mass'
nvars = 3

y_train_label=in_df['isSignal'].values[:nevt] #[0 1 0 ...]
y_test_label=in_df['isSignal'].values[nevt:2*nevt] #same
x_train=in_df.loc[:nevt-1,[var1, var2, var3]].values #[[var1 var2 var3], [var1 var2 var3],...]
x_test=in_df.loc[nevt:2*nevt-1,[var1, var2, var3]].values
y_train=np.eye(2)[y_train_label]
y_test=np.eye(2)[y_test_label]

trainDict={"signal": [], "background": []}
testDict ={"signal": [], "background": []}


label_names = ['background','signal']
plotVars(x_train, in_df, [var1,var2,var3], label_names)
plotTruth(x_train,in_df,label_names)

#TODO better way of dealing with this?
for i in range(0,nevt):
    if (y_train_label[i]==1):
        trainDict["signal"].append(x_train[i].tolist())
    else:
         trainDict["background"].append(x_train[i].tolist())
trainDict={"signal": np.array(trainDict["signal"]), "background":  np.array(trainDict["background"])}
            
for i in range(0,nevt):
    if (y_test_label[i]==1):
        testDict["signal"].append(x_test[i].tolist())
    else:
         testDict["background"].append(x_test[i].tolist())
testDict={"signal": np.array(testDict["signal"]), "background":  np.array(testDict["background"])}

print("Test and Training data ready")

backend = BasicAer.get_backend('qasm_simulator')

#configure optimizer
#max_trials=1000,
#save_steps=1
#last_avg=1
#c0=0.6283185307179586
#c1=0.1
#c2=0.602
#c3=0.101
#c4=0
#skip_calibration=False
#Args:
#    max_trials (int): Maximum number of iterations to perform.
#    save_steps (int): Save intermediate info every save_steps step.
#    last_avg (int): Averaged parameters over the last_avg iterations.
#                    If last_avg = 1, only the last iteration is considered.
#    c0 (float): The initial a. Step size to update parameters.
#    c1 (float): The initial c. The step size used to approximate gradient.
#    c2 (float): The alpha in the paper, and it is used to adjust a (c0) at each iteration.
#    c3 (float): The gamma in the paper, and it is used to adjust c (c1) at each iteration.
#    c4 (float): The parameter used to control a as well.
#    skip_calibration (bool): skip calibration and use provided c(s) as is.
optimizer = SPSA(
				max_trials=int(args.maxTrials),
				c0=4.0,
				save_steps=int(args.saveSteps),
				skip_calibration=True
			)

#configure feature map
#feature_dimension,
#depth=2,
#entangler_map=None,
#entanglement='full', 
#data_map_func=<function self_product>
#Args:
#    feature_dimension (int): number of features
#    depth (int): the number of repeated circuits
#    entangler_map (list[list]): describe the connectivity of qubits, each list describes
#                                [source, target], or None for full entanglement.
#                                Note that the order is the list is the order of
#                                applying the two-qubit gate.
#    entanglement (str): ['full', 'linear'], generate the qubit connectivity by predefined
#                        topology
#    data_map_func (Callable): a mapping function for data x
feature_map = SecondOrderExpansion(
	feature_dimension=3, 
	depth=int(args.featMapDepth),
	entanglement = 'full'
)

#variational form to take
# Args:
#  num_qubits (int) : number of qubits
#  depth (int) : number of rotation layers
#  entangler_map (list[list]): describe the connectivity of qubits, each list describes
#                              [source, target], or None for full entanglement.
#                              Note that the order is the list is the order of
#                              applying the two-qubit gate.
#  entanglement (str): 'full' or 'linear'
#  initial_state (InitialState): an initial state object
#  entanglement_gate (str): cz or cx
#  skip_unentangled_qubits (bool): skip the qubits not in the entangler_map
var_form = RYRZ(
	num_qubits=3,
	depth=int(args.varFormDepth),
	entanglement='full',
	entanglement_gate='cz',
)

#create VQC
#Args:
#    optimizer (Optimizer): The classical optimizer to use.
#    feature_map (FeatureMap): The FeatureMap instance to use.
#    var_form (VariationalForm): The variational form instance.
#    training_dataset (dict): The training dataset, in the format:
#                            {'A': np.ndarray, 'B': np.ndarray, ...}.
#    test_dataset (dict): The test dataset, in same format as `training_dataset`.
#    datapoints (np.ndarray): NxD array, N is the number of data and D is data dimension.
#    max_evals_grouped (int): The maximum number of evaluations to perform simultaneously.
#    minibatch_size (int): The size of a mini-batch.
#    callback (Callable): a callback that can access the
#        intermediate data during the optimization.
#        Internally, four arguments are provided as follows the index
#        of data batch, the index of evaluation,
#        parameters of variational form, evaluated value.
vqc = VQC(optimizer, feature_map, var_form, trainDict, testDict)

#a quantum instance
#Args:
#    backend (BaseBackend): instance of selected backend
#    shots (int, optional): number of repetitions of each circuit, for sampling
#    seed_simulator (int, optional): random seed for simulators
#    max_credits (int, optional): maximum credits to use
#    basis_gates (list[str], optional): list of basis gate names supported by the
#                                       target. Default: ['u1','u2','u3','cx','id']
#    coupling_map (CouplingMap or list[list]): coupling map (perhaps custom) to
#                                              target in mapping
#    initial_layout (Layout or dict or list, optional): initial layout of qubits in mapping
#    pass_manager (PassManager, optional): pass manager to handle how to compile the circuits
#    seed_transpiler (int, optional): the random seed for circuit mapper
#    optimization_level (int, optional): How much optimization to perform on the circuits.
#                                        Higher levels generate more optimized circuits,
#                                        at the expense of longer transpilation time.
#    backend_options (dict, optional): all running options for backend, please refer
#                                      to the provider.
#    noise_model (qiskit.provider.aer.noise.noise_model.NoiseModel, optional): noise model
#                                                                              for simulator
#    timeout (float, optional): seconds to wait for job. If None, wait indefinitely.
#    wait (float, optional): seconds between queries to result
#    circuit_caching (bool, optional): Use CircuitCache when calling compile_and_run_circuits
#    cache_file(str, optional): filename into which to store the cache as a pickle file
#    skip_qobj_deepcopy (bool, optional): Reuses the same Qobj object
#                                         over and over to avoid deepcopying
#    skip_qobj_validation (bool, optional): Bypass Qobj validation to
#                                            decrease submission time
#    measurement_error_mitigation_cls (Callable, optional): the approach to mitigate
#                                                            measurement
#                                                            error, CompleteMeasFitter or
#                                                            TensoredMeasFitter
#    cals_matrix_refresh_period (int, optional): how long to refresh the calibration
#                                                matrix in measurement mitigation,
#                                                unit in minutes
#    measurement_error_mitigation_shots (int, optional): the shot number for building
#                                                        calibration matrix,
#                                                        if None, use the shot number
#                                                        in quantum instance
#    job_callback (Callable, optional): callback used in querying info of
#                                       the submitted job, and
#                                       providing the following arguments: job_id,
#                                       job_status, queue_position, job
quantum_instance = QuantumInstance(
	backend = backend,
 	shots=int(args.numberShots),
	seed_simulator=420,
	seed_transpiler=420,
)

print("Running Algorithm")
if not args.steerTestRun:
    result = vqc.run(quantum_instance)
    print("testing success ratio: ", result['testing_accuracy'])
#    print("predicted classes:", result['predicted_classes'])

#time or tag setting in name
outtag="_".join([str(vars(args)[i]) if not "steer" in str(i) else "" for i in vars(args)])
outtag+="_%s"%(int(time.time()))
pklFile=open("{0}/qicl_test_{1}.pkl".format(args.steerOutDir,outtag),'wb')
pickle.dump( result , pklFile)
pickle.dump( vars(args) , pklFile)

m, s = divmod(time.time()-time0, 60)
h, m = divmod(m, 60)
print("Execution time: %d:%02d:%02d" % (h,m,s))
print("Success!")
