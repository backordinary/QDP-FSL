# https://github.com/marklong-pg/structural-encoding/blob/7b2a9fa61191be94e9d27bb2c574db888fc4a6d6/utils.py
from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance
from qiskit.visualization import plot_histogram
from sklearn.metrics import roc_curve
from data_extraction import get_X_y, get_subset_n, get_training_test_data
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.base import clone as clone_base
import numpy as np
matlab_data_dir = "C:/Users/DELL/MATLAB Drive/Research 2022/Forrelation Datageneration/" \
                  "Test Ground/samplingByLayers/Datasets/"

def execute_qasm(quantum_circuit, shots=8096):
    cir_meas = quantum_circuit.measure_all(inplace=False)
    qinst = QuantumInstance(backend=Aer.get_backend('qasm_simulator'), shots=shots)
    result = qinst.execute(cir_meas).get_counts()
    display(plot_histogram(result))
    return result

def draw_learning_curves_increasing_k(filename, training_sizes, ax, color='red', n_lines=-1, error_bar=False, is_repeat=False):
    """
    Draw learning curves of increasing k given a filename
    Args:
        filename: str, path to result file from current directory
        training_sizes: np.array, list of training sizes used in the learning curve
        ax: matplotlib.axes, axis for plotting
        n_lines: int, number of lines to draw
        color: str, color for the plot
        error_bar: boolean, whether to plot the error bar
        is_repeat: boolean, whether the data is of the same n and k
    """
    data = np.loadtxt(fname=filename,delimiter=',')
    if not is_repeat:
        mean_vals = data[0::2]
        std_vals = data[1::2]
    else:
        vals = data[0::2]
        mean_vals = vector_to_matrix(np.mean(vals,axis=0))
        std_vals = vector_to_matrix(np.std(vals,axis=0))
    if n_lines == -1: n_lines = mean_vals.shape[0]
    transparency_vals = np.linspace(1, 0.3, n_lines)
    for row in range(mean_vals.shape[0]):
        alpha = transparency_vals[row]
        ax.plot(training_sizes, mean_vals[row],color=color,alpha=alpha)
        if error_bar:
            ax.fill_between(training_sizes,mean_vals[row]-std_vals[row],mean_vals[row]+std_vals[row],color=color,alpha=alpha/2)
        if row + 1 == n_lines:
            break
    # ax.hlines(0.5,training_sizes[0],training_sizes[-1],colors='blue',linestyles='dashed')
    # ax.set_xlabel('Training Size')
    # ax.set_ylabel('Average Accuracy')
    return ax

def vector_to_matrix(vector):
    return np.reshape(vector,[1, vector.shape[0]])

def get_performances(filename):
    """Get final classification performances in k from result file"""
    data = np.loadtxt(fname=filename,delimiter=',')
    mean_vals = data[0::2][:,-1].T
    std_vals = data[1::2][:,-1]
    return mean_vals

def get_kernel_matrix(filename,kernel=RBF()):
    """Draw kernel matrix for the dataset of the given filename"""
    X, y = get_X_y(file_name=filename)
    X, y, _, _ = get_subset_n(X,y,100)
    return kernel(X)






