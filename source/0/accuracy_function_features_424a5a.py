# https://github.com/blatmand/blatmandThesis/blob/f29812a060f490da80f9395deec12e70668f28e2/ClassifyGW/files_all_three_observation_runs/accuracy_function_features.py
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:43:36 2022

@author: Daniel Blatman
This file run classical and quantum support vector machines and compares
the run time as a function of features as well as the accuracy
as a function of features. The training events use measurement noise
from before the event GW150914. The feature extraction step uses
the library TSFEL, where all temporal, all statistical and the wavelet features
which are a subset of the spectral features are used.
The script plots the average balanced accuracy for the test 
events 'GW150914','GW151012','GW151226','GW170814','GW170823','GW170729',
'GW170104','GW170608','GW170809','GW170817','GW170818' as a function of the
used features. As the covariant feature map needs an even number of features,
a redundant feature in the form of a copy of the last feature is added to the 
features used for classification for both the classical and quantum SVM for 
cases where the number of features is odd.
The script saves the figures:
"qubit_dependece_temp_and_stat_and_wavelet_features_classical_quantum"
"algorithm_scaling_temp_and_stat_features_classical_quantum"

"""
import time
from MySVCLoss import MySVCLoss
masses_dict = {
        'GW150914': [35.6,30.6],
        'GW151012': [23.2,13.6],
        'GW151226': [13.7,7.7],
        'GW170814': [30.6,25.2],
        'GW170823': [39.5,29.0],
        'GW170729': [50.2,34.0],
        'GW170104': [30.8,20.0],
        'GW170608': [11.0,7.6],
        'GW170809': [35.0,23.8],
        'GW170817': [1.46,1.27],
        'GW170818': [35.4,26.7]
        }

from qiskit.utils import algorithm_globals

seed = 12345
algorithm_globals.random_seed = seed
from qiskit import BasicAer 
from qiskit_machine_learning.kernels import QuantumKernel
import os,sys
import matplotlib.pyplot as plt
# Put this repository on the Python path and import qkt pkgs
module_path = os.path.abspath(os.path.join('../../prototype-quantum-kernel-training'))
sys.path.append(module_path)
from qkt.feature_maps import CovariantFeatureMap
from qkt.utils import QKTCallback
#from qiskit.visualization import circuit_drawer
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.algorithms import QSVC
# process raw data from Ligo
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from scipy import signal
import numpy as np
import tsfel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics

def get_label(index1,index2,long_data_test,gps):
    t1=long_data_test.xindex[index1].value
    t2=long_data_test.xindex[index2].value
    if(gps >= t1 and gps <= t2):
        return 1
    return -1
attribute = None

cfg_file = tsfel.get_features_by_domain(attribute)
  
for key in cfg_file['spectral'].keys():
    cfg_file['spectral'][key]['use']='no'  
cfg_file['spectral']['Wavelet energy']['use']='yes'
cfg_file['spectral']['Wavelet entropy']['use']='yes'
cfg_file['spectral']['Wavelet standard deviation']['use']='yes'
cfg_file['spectral']['Wavelet variance']['use']='yes'
cfg_file['spectral']['Wavelet absolute mean']['use']='yes'
                     
events=['GW150914','GW151012','GW151226','GW170814','GW170823'
        ,'GW170729','GW170104','GW170608','GW170809','GW170817'
        ,'GW170818']
def create_filename(i,j,file_id,train_event):
    filename = "gwpy_hc_mass1_"+str(i)+"_mass2_"+str(j)+"real_data_noise_"+file_id+"_from_"+train_event+"_.txt"
    return filename 
def process_event(event_id):
    gps = event_gps(event_id)
    segment = (int(gps) - 2, int(gps) + 10)
    hdata = TimeSeries.fetch_open_data('H1', *segment, verbose=True, cache=True)

    data_bp = hdata.bandpass(50, 250, filtfilt=True)
    data_bp = data_bp.crop(int(gps)- 1, int(gps) +9)

    notches = [filter_design.notch(f, data_bp.sample_rate) for f in (60, 120, 180)]
    powernotch = filter_design.concatenate_zpks(*notches)
    hclean = data_bp.filter(powernotch, filtfilt=True)
    long_data_test = hclean
    y_test_long_signal=[]
    fs= int(hclean.sample_rate.value)
    start_index = 0
    end_index = int(fs/4)
    data_segments_test = []
    # split data into 0.25 windows with 0.125 seconds overlap
    test_window = signal.windows.tukey(long_data_test[start_index:int(end_index)].size)
    for i in range(79):
        data_test= long_data_test[start_index:end_index]
        data_test_window = data_test * test_window
        y_test_long_signal.append(get_label(start_index,end_index-1,long_data_test,gps)) 
        start_index = start_index + int(fs/8)
        end_index = end_index +int(fs/8)
        data_segments_test.append(data_test_window)
    x_test_long_signal = data_segments_test
    return hclean, gps, fs, x_test_long_signal, y_test_long_signal

filename_ids = ["s0.001"]
legend=["1E-3"]
num_features_used=[]
total_classical_times=[]
total_quantum_times=[]
classical_total=[]
quantum_total=[]
for file_id in filename_ids:
    quantum_av_balanced_accuracies=[]
    av_balanced_accuracies = []
    results_hclean = []
    results_gps = []
    results_fs= []
    results_x_test_long_signal =[]
    results_y_test_long_signal = []
    for i in range(len(events)):
        hclean, gps, fs, x_test_long_signal, y_test_long_signal = process_event(events[i])
        results_hclean.append(hclean)
        results_gps.append(gps)
        results_fs.append(fs)
        results_x_test_long_signal.append(x_test_long_signal)
        results_y_test_long_signal.append(y_test_long_signal)
    
    hclean1 = results_hclean[0]
    # using the time series feature extraction library tsfel
    data_segments_train = []
    filenames_train=[]
    train_events = ['GW150914'] 
    for iii in range(len(train_events)):
        data_train = TimeSeries.read('different_noise_long_training_data_noise_bp_and_notch_filtered_'+train_events[iii]+'.txt')
        length = 25.0
        
        fs = int(hclean1.sample_rate.value)
    
        start_index = 0
        end_index = int(fs*0.25)
        train_window = signal.windows.tukey(data_train[start_index:int(end_index)].size)
    
        for k in range(int(length*4)):
            data_segment_train= data_train[start_index:int(end_index)]
            hwin_train = data_segment_train * train_window
            start_index = start_index + int(0.25*fs)
            end_index = end_index +int(0.25*fs)
            data_segments_train.append(hwin_train)
        
    num=10
    for train_event in train_events:
        for i in range(num):
            for j in range(num):
                filenames_train.append(create_filename(31+i,23+j,file_id,train_event))
        
    names_train = filenames_train        


    total_train_data = []
    train_labels=[]

    windowed_total_train_data=[]

    for i in range(len(names_train)):
        read_data = TimeSeries.read(names_train[i])
        total_train_data.append(read_data)

    for i in range(len(data_segments_train)):
        total_train_data.append(data_segments_train[i])  

    windowed_simulated_data_array = []
    simulated_data_window = signal.windows.tukey(len(total_train_data[0]))
    for j in range(len(names_train)):
    
        windowed_simulated_data = total_train_data[j]*simulated_data_window
        windowed_simulated_data_array.append(windowed_simulated_data)

    for i in range(len(windowed_simulated_data_array)):
        windowed_total_train_data.append(windowed_simulated_data_array[i])
        train_labels.append(1)

    for i in range(len(data_segments_train)):
        windowed_total_train_data.append(data_segments_train[i])  
        train_labels.append(-1)

    x_train_sig = windowed_total_train_data
    X_train = tsfel.time_series_features_extractor(cfg_file, x_train_sig, fs=fs)

    results_X_test_long_signal=[]
    for i in range(len(events)):
        results_X_test_long_signal.append(tsfel.time_series_features_extractor(cfg_file, results_x_test_long_signal[i], fs=results_fs[i]))

    # Highly correlated features are removed
    corr_features = tsfel.correlated_features(X_train)
    X_train.drop(corr_features, axis=1, inplace=True)
    tempvars=[]
    for i in range(len(events)):
        tempvars.append(results_X_test_long_signal[i])
        results_X_test_long_signal[i].drop(corr_features, axis=1, inplace=True)


    # Remove low variance features
    selector = VarianceThreshold()
    tempvars2=[]
    for i in range(len(events)):
        tempvars2.append(results_X_test_long_signal[i])
    X_train = selector.fit_transform(X_train)
    for i in range(len(events)):
        results_X_test_long_signal[i] = selector.transform(results_X_test_long_signal[i])

    # Normalising Features
    scaler = preprocessing.StandardScaler()
    nX_train = scaler.fit_transform(X_train)
    results_nX_test_long_signal = []
    for i in range(len(events)):
        results_nX_test_long_signal.append(scaler.transform(results_X_test_long_signal[i]))
    num_columns = nX_train[:,:].shape[1]   
    # copy last feature for an odd number of features
    if(num_columns % 2 == 1):
        last_column=nX_train[:,-1]
        last_column.resize(last_column.shape[0]  ,1)
        nX_train=np.hstack((nX_train,last_column))
        for i in range(len(events)):
            temp_result = results_nX_test_long_signal[i]
            last_column_temp= temp_result[:,-1]  
            last_column_temp.resize(last_column_temp.shape[0]  ,1)
            results_nX_test_long_signal[i]=np.hstack((temp_result,last_column_temp))    
        
    num_columns = nX_train[:,:].shape[1]     
    num_features_used.append(num_columns)
    
    classical_times=[]
    quantum_times=[]
    
    while(num_columns>=2):
        
        print("***************************************")
        print("number of columns is: "+str(num_columns)) 
        print("***************************************")
        total_balanced_accuracies=[]
        nX_train = nX_train[:,0:(num_columns)]
        for i in range(len(events)):
            temp_result = results_nX_test_long_signal[i]
            results_nX_test_long_signal[i] = temp_result[:,0:(num_columns)]
        y_train = train_labels 
        
        # start the classical algorithm part
        tic = time.perf_counter()
        
        C=2
        gamma=1
        model=SVC(C=C,gamma=gamma,kernel='rbf')
        model.fit(nX_train,y_train)
        results_labels_test_long_signal = []
        results_accuracy_test_svc_long_signal = []
        for i in range(len(events)):
            results_labels_test_long_signal.append(model.predict(results_nX_test_long_signal[i]))
            results_accuracy_test_svc_long_signal.append(metrics.balanced_accuracy_score(y_true=results_y_test_long_signal[i], y_pred=results_labels_test_long_signal[i]))
    
       
        av_balanced_accuracies.append(sum(results_accuracy_test_svc_long_signal)/len(results_accuracy_test_svc_long_signal))
        toc = time.perf_counter()
        classical_times.append(toc-tic)
        print("elapsed time for the classical algorithm is:"+str(toc-tic))
        
        
        # start the quantum algorithm part
        tic = time.perf_counter()
        seed = 12345
        
        algorithm_globals.random_seed = seed
        
        num_features = np.shape(nX_train)[1]
        entangler_map =[]
        for index in range(int(num_features/2)-1):
            entangler_map.append([index,index+1])
        
        fm = CovariantFeatureMap(
            feature_dimension=num_features,
            entanglement=entangler_map,
            single_training_parameter=False
        )


        # Use the qasm simulator backend
        backend = BasicAer.get_backend('statevector_simulator')
        import sys

        labels = np.array(y_train)
        y_test_long_signal = np.array(y_test_long_signal)
        # Instantiate quantum kernel
        quant_kernel = QuantumKernel(fm,
                                     user_parameters=fm.user_parameters,
                                     quantum_instance=backend)
        learning_rate=0.02
        # Set up the optimizer
        cb_qkt = QKTCallback()
        spsa_opt = SPSA(maxiter=300,
                        callback=cb_qkt.callback,
                        learning_rate=learning_rate,
                        perturbation=0.02
                   )
        # Instantiate a quantum kernel trainer.
        qkt = QuantumKernelTrainer(
            quantum_kernel=quant_kernel,
            #loss="svc_loss",
            loss=MySVCLoss(C=C),
            optimizer=spsa_opt,
            initial_point=[0.1]*len(fm.user_parameters)
        )
        # Train the kernel using QKT directly
        qka_results = qkt.fit(nX_train, y_train)
        optimized_kernel = qka_results.quantum_kernel

        # Use QSVC for classification
        qsvc = QSVC(C=C,quantum_kernel=optimized_kernel)
        
        # Fit the QSVC
        qsvc.fit(nX_train, y_train)                 
        accuracies_qka=[]

        # test the SVM on new data:
        results_labels_test_long_signal_qsvc = []
        results_accuracy_test_svc_long_signal_qsvc = []
        for i in range(len(events)):
            results_labels_test_long_signal_qsvc.append(qsvc.predict(results_nX_test_long_signal[i]))
            results_accuracy_test_svc_long_signal_qsvc.append(metrics.balanced_accuracy_score(y_true=results_y_test_long_signal[i], y_pred=results_labels_test_long_signal_qsvc[i]))

        
        quantum_av_balanced_accuracies.append(sum(results_accuracy_test_svc_long_signal_qsvc)/len(results_accuracy_test_svc_long_signal_qsvc))
        
        toc = time.perf_counter()
        quantum_times.append(toc-tic)
        print("elapsed time for the quantum algorithm is:"+str(toc-tic))
        
        num_columns=num_columns-2   
    classical_total.append(av_balanced_accuracies)
    quantum_total.append(quantum_av_balanced_accuracies)
    total_classical_times.append(classical_times)
    total_quantum_times.append(quantum_times)
    
#calculate average accuracy for each file id:  
features=[]

for i in range(len(classical_total)):
    features.append(list(reversed(np.arange(2,num_features_used[i]+2,step=2))))
plt.figure(figsize=(12,6))
for i in range(len(classical_total)):
    plt.plot(features[i], classical_total[i],'.', label="classical: s = "+legend[i],color="red")
for i in range(len(quantum_total)):
    plt.plot(features[i], quantum_total[i],'.', label="quantum: s = "+legend[i],color="green")    
plt.title("Number of features dependency")
plt.ylabel("average balanced accuracy")
plt.xlabel("features used")
plt.legend()
plt.savefig("qubit_dependece_temp_and_stat_and_wavelet_features_classical_quantum")
plt.close()


features=[]
for i in range(len(classical_total)):
    features.append(list(reversed(np.arange(2,num_features_used[i]+2,step=2))))

plt.figure(figsize=(12,6))
for i in range(len(classical_total)):
    plt.plot(features[i], classical_times,'.', label="classical: s = "+legend[i],color="red")
for i in range(len(quantum_total)):
    plt.plot(features[i], quantum_times,'.', label="quantum: s = "+legend[i],color="green")   
plt.title("Quantum and classical algorithm")
plt.ylabel("time[s]")
plt.xlabel("features used")
plt.yscale("log")
plt.legend()
plt.savefig("temp_and_stat_and_wavelet_features_classical_quantum")
plt.close()
import math
log_quantum_times=[]
for i in range(len(total_quantum_times[0])):
    log_quantum_times.append(math.log2(total_quantum_times[0][i]))      
polynomial = np.polyfit(features[0],log_quantum_times , 1)
p=np.poly1d(polynomial)
x=np.linspace(1,26,1000)    

plt.figure(figsize=(12,6))
for i in range(len(quantum_total)):
    plt.plot(features[i], log_quantum_times,'.', label="log base 2 of quantum times: s = "+legend[i],color="green")   

plt.plot(x, p(x),label="linear fit", linewidth=0.5, color="blue")    
plt.title("Quantum algorithm scaling")
plt.ylabel("log base 2 of quantum times")
plt.xlabel("features used")
plt.ylim(0, 20)
plt.legend(linewidth=0.5)
plt.savefig("log_base_2_quantum_times_algorithm_scaling_temp_and_stat_and_wavelet_features")
plt.close()
