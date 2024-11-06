# https://github.com/VoicuTomut/QMap/blob/73bd68f3ca426b33f9a4dce709a317a23dd3af58/project_qmap/tsp_vqe.py
from qiskit import BasicAer, Aer, IBMQ
from qiskit.circuit import QuantumCircuit,QuantumRegister, ParameterVector

from qiskit.tools.visualization import plot_histogram
from qiskit.optimization.applications.ising import tsp
from qiskit.aqua.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.aqua.components.optimizers import COBYLA

# not n
from qiskit.optimization.problems import QuadraticProgram


from qiskit.optimization.algorithms import MinimumEigenOptimizer
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())



def cost(rut, w):
    '''

    :param rut:
    :param w:
    :return:
    '''
    dist = 0
    for i in range(0, len(rut) - 1):
        dist = dist + w[rut[i]][rut[i + 1]]
    dist = dist + w[rut[i + 1]][rut[0]]

    return dist


class jojo:
    def __init__(self, w):
        '''
        :param w:
        '''
        self.w = w
        self.dim = len(w[0])

