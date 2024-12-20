# https://github.com/pafloxy/QuantumAlgorithmsCollect/blob/ceb91b95d7c0a126b70d02ca20e6cfdea2e5bcef/qiskit_utils.py
## imports
from qiskit import *
from qiskit.algorithms import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import quantum_info, IBMQ, Aer
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
# backend = BasicAer.get_backend("statevector_simulator")
# quantum_instance = QuantumInstance(backend)
from qiskit.algorithms import AmplitudeEstimation
from qiskit.circuit.library import RYGate ,SwapGate, CSwapGate
#from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_bloch_multivector, plot_bloch_vector
import numpy as np
from numpy import pi
import seaborn as sns
from IPython.display import Image
import matplotlib.pyplot as plt
from typing import Optional, Union, Iterable


# provider = IBMQ.load_account()
# IBMQ.get_provider(hub='ibm-q-education', group='iit-madras-1', project='quantum-computin')
# # setup required backends 
# lima = provider.get_backend('ibmq_lima')
# manila = provider.get_backend('ibmq_manila')
qsm = Aer.get_backend('qasm_simulator')
stv = Aer.get_backend('statevector_simulator')
aer = Aer.get_backend('aer_simulator')


##################################################################################################
                                    ## helper functions ##
##################################################################################################

def bit_conditional( num: int, qc: QuantumCircuit, cntrl: QuantumRegister): ## @helper_function
    """ helper function to assist the conditioning of ancillas 
    
        ARGS:
        ----
            num : the integer to be conditioned upon 
            qc : QuantumCircuit of the original circuit
            register : QuantumRegister corresponding to the control_ancilla which is conditioned upon 
        
        RETURNS:
        -------
            Works inplace i.e modifies the original circuit 'qc' that was passed """

    n = len(cntrl)
    bin_str = format(num, "b").zfill(n)
    assert len(bin_str) == len(cntrl)            ## need to assurethe bit ordering covention according tto qiskit's

    bin_str = bin_str[::-1]                         ## reverse the bit string
    for index, bit in enumerate(bin_str):
        if bit== '0': qc.x(cntrl[index])



def measure_and_plot(qc: QuantumCircuit, shots:int= 1024, show_counts:bool= False, return_counts:bool= False, measure_cntrls: bool = False, decimal_count_keys:bool = True , cntrl_specifier:Union[int, list, str] = 'all'):
    """ Measure and plot the state of the data registers, optionally measure the control ancillas, without modifying the original circuit.
        
        ARGS:
        ----
            qc : 'QuantumCircuit' 
                 the circuit to be measured

            shots: 'int' 
                    no. of shots for the measurement

            show_counts : 'bool' 
                           print the counts dictionary

            measure_cntrls : 'bool' 
                             indicates whether to measure the control ancilla registers.
            
            return_counts : 'bool'
                            returns the counts obtained if True, else retruns the histogram plot

            measure_cntrls: 'bool'
                             indicates whether to measure the controll ancill qubits
                
            decimal_count_keys: 'bool'
                                if 'True' converts the binary state of the controll ancilllas to integer represntation

            cntrl_specifier : 'int' 
                                inidicates whihch of the control registers to meausure, 
                                for eg. cntrl_specifier= 2 refers to the first control ancilla cntrl_2
                                cntrl_specifier= 'all' refers to all the ancillas                                                           
        RETURNS:
        -------
            plots histogram over the computational basis states

     """
    qc_m = qc.copy()
    creg = ClassicalRegister( len(qc_m.qregs[0]) )
    qc_m.add_register(creg)
    qc_m.measure(qc_m.qregs[0], creg)

    if measure_cntrls== True:

        if isinstance(cntrl_specifier, int):
            print('int')##cflag
            if cntrl_specifier > len(qc_m.qregs) or cntrl_specifier < 1: raise ValueError(" 'cntrl_specifier' should be less than no. of control registers and greater than 0")

            creg_cntrl = ClassicalRegister(len(qc_m.qregs[cntrl_specifier-1]))
            qc_m.add_register(creg_cntrl)
            qc_m.measure(qc_m.qregs[cntrl_specifier-1], creg_cntrl )

        elif isinstance(cntrl_specifier, list ):
            print('list')##cflag
            for ancilla in cntrl_specifier:

                if ancilla > len(qc_m.qregs) or ancilla < 1: raise ValueError(" 'ancilla' should be less than no. of control registers and greater than 0")

                creg_cntrl = ClassicalRegister(len(qc_m.qregs[ancilla-1]))
                qc_m.add_register(creg_cntrl)
                qc_m.measure(qc_m.qregs[ancilla-1], creg_cntrl )

        elif isinstance(cntrl_specifier, str) and cntrl_specifier== "all":
            print('str')##cflag
            for reg in qc_m.qregs[1:] :
                creg = ClassicalRegister(len(reg))
                qc_m.add_register(creg)
                qc_m.measure(reg, creg)
                
    # plt.figure()
    counts = execute(qc_m, qsm, shots= shots).result().get_counts()
   
    if decimal_count_keys:
        counts_m = {}
        for key in counts.keys():
            split_key = key.split(' ')
            if measure_cntrls:
                key_m = 'key: '
                for string in split_key[:-1]:
                    key_m+= str(int(string, 2)) + ' '
                key_m += '-> '
            else: key_m = ''
            key_m += split_key[-1][::-1] ##to-check
            counts_m[key_m] = counts[key]
        counts = counts_m

    if show_counts== True: print(counts)
    
    if return_counts: return counts
    else: return plot_histogram(counts)
    

   

####################################################################################################
