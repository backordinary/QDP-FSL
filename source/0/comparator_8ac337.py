# https://github.com/aymeric-mcrae/QQML/blob/94fdf2b752076cd05485041677702377ca7634d6/Code/comparator.py
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from IPython.display import HTML, display
import tabulate

import numpy as np
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM


from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA

from sklearn.model_selection import train_test_split

import time

from qiskit.aqua.components.multiclass_extensions import AllPairs
from qiskit.aqua.algorithms.classifiers.qsvm._qsvm_estimator import _QSVM_Estimator




from qiskit.circuit.library import ZZFeatureMap, TwoLocal


def checkAllSame(arr):
    first = arr[0]
    for a in arr:
        if (a != first):
            return False
    return True

def convertFromQS(training_data, test_data):
    X_train = [x for x in training_data["A"]]+([x for x in training_data["B"]])
    x_test = [x for x in test_data["A"]]+([x for x in test_data["B"]])
    Y_train = [0 for x in training_data["A"]]+([1 for x in training_data["B"]])
    y_test = [0 for x in test_data["A"]]+([1 for x in test_data["B"]])
    
    try:
        X_train = X_train + [x for x in training_data["C"]]
        x_test = x_test + [x for x in test_data["C"]]
        Y_train = Y_train + [2 for x in training_data["C"]]
        y_test = y_test + [2 for x in test_data["C"]]
    except:
        kk=1 #Hacky workaround
    
    return (X_train, x_test, Y_train, y_test)

#All-in-one tester method, generates table of results
def compareMethods(class1, class2, class3 = None, backend=BasicAer.get_backend('qasm_simulator'), name = "", 
                   include_unscaled=False, include_QSVM = True, include_VQC = True, feature_dimension = 2, gamma = 'auto', C = 1.0):
  
    #Define header and chart data
    data = []
    header = ["Algorithm", "Backend", "Time", "Accuracy", "Only one Class Predicted?"]
    data.append(header)
    
    #Split data into train and test
    class1_train, class1_test = train_test_split(class1, test_size=0.33, random_state=42)
    class2_train, class2_test = train_test_split(class2, test_size=0.33, random_state=42)
    feature_dim = feature_dimension
    if class3 is not None:
        class3_train, class3_test = train_test_split(class3, test_size=0.33, random_state=42)

    #Get input data for quantum
    training_data = {'A': np.asarray(class1_train), 'B': np.asarray(class2_train)}
    test_data = {'A': np.asarray(class1_test), 'B': np.asarray(class2_test)}
    total_array = np.concatenate((test_data['A'], test_data['B']))
    
    if class3 is not None:
        training_data["C"] = class3_train
        test_data["C"] = class3_test
        total_array = np.concatenate((total_array, test_data['C']))

    
    #Get input data for classical
    X_train, x_test, Y_train, y_test = convertFromQS(training_data, test_data)

    #Classical SVM, linear kernel (scaled and unscaled)
    if include_unscaled:
        start = time.time()
        clf = svm.SVC(kernel='linear') # Linear Kernel
        model = clf.fit(X_train, Y_train)
        y_pred = clf.predict(x_test)
        end = time.time()
        data.append(["SVM, Linear Kernel", "Local Processor", round(end-start), str(round(100*metrics.accuracy_score(y_test, y_pred), 2)),checkAllSame(y_pred)])
    
    start = time.time()
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    x_test_std = scaler.fit_transform(x_test)
    clf = svm.SVC(kernel='linear') # Linear Kernel
    model = clf.fit(X_train_std, Y_train)
    y_pred = clf.predict(x_test_std)
    end = time.time()
    data.append(["SVM, Linear Kernel, scaled", "Local Processor", round(end-start), str(round(100*metrics.accuracy_score(y_test, y_pred), 2)),checkAllSame(y_pred)])
        
    #Classical SVM, rbf kernel (scaled and unscaled)
    if include_unscaled:
        start = time.time()
        clf = svm.SVC(C=C, kernel='rbf', gamma = gamma) # rbf Kernel
        model = clf.fit(X_train, Y_train)
        y_pred = clf.predict(x_test)
        end = time.time()
        data.append(["SVM, RBF Kernel", "Local Processor", round(end-start), str(round(100*metrics.accuracy_score(y_test, y_pred), 2)),checkAllSame(y_pred)])

    start = time.time()
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    x_test_std = scaler.fit_transform(x_test)
    clf = svm.SVC(C=C, kernel='rbf', gamma = gamma) # rbf Kernel
    model = clf.fit(X_train_std, Y_train)
    y_pred = clf.predict(x_test_std)
    end = time.time()
    data.append(["SVM, RBF Kernel, scaled", "Local Processor", round(end-start), str(round(100*metrics.accuracy_score(y_test, y_pred), 2)),checkAllSame(y_pred)])

    
    #QSVM run
    if include_QSVM:
        start = time.time()
        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')
        if class3 is None:
            qsvm = QSVM(feature_map, training_data, test_data, total_array)
        else:
            qsvm = QSVM(feature_map, training_data, test_data, total_array, multiclass_extension=AllPairs())           
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=10598, seed_transpiler=10598)
        resultSVM = qsvm.run(quantum_instance)
        end = time.time()
        QSVM_Summary = ["QSVM", backend.name(), round(end-start), str(round(100*resultSVM['testing_accuracy'], 2)), checkAllSame(resultSVM['predicted_classes'])]
        data.append(QSVM_Summary)
        path = 'C:\\Users\\admin\\Desktop\\QQML\\Code\\Saved_SVMs\\' + name + "_" + backend.name() + "_QSVM"
        if class3 is None: #Bug in package prevents saving Multiclass svms. Will find workaround or submit bug report if time.
            qsvm.save_model(path)
    
    #VQC run
    if include_VQC:
        start = time.time()
        optimizer = SPSA(max_trials=100, c0=4.0, skip_calibration=True)
        optimizer.set_options(save_steps=1)
        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
        var_form = TwoLocal(feature_dim, ['ry', 'rz'], 'cz', reps=3)
        vqc = VQC(optimizer, feature_map, var_form, training_data, test_data, total_array)
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=10589, seed_transpiler=10598)
        resultVQC = vqc.run(quantum_instance)
        end = time.time()
        VQC_Summary = ["VQC", backend.name(), round(end-start), str(round(100*resultVQC['testing_accuracy'], 2)), checkAllSame(resultVQC['predicted_classes'])]
        data.append(VQC_Summary)
        path = 'C:\\Users\\admin\\Desktop\\QQML\\Code\\Saved_SVMs\\' + name + "_" + backend.name() + "_VQC"
        vqc.save_model(path)
    
    display(HTML(tabulate.tabulate(data, tablefmt='html')))
    return data