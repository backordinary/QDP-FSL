# https://github.com/immoth/Hubbard_symmetries/blob/44339d4d7062ce5fac8f51e0580289296551e07f/Gutzwiller/Object_Oriented/Define_Quantum_Device.py

from qiskit import IBMQ,transpile
from qiskit.tools.monitor import job_monitor
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-afrl', group='air-force-lab', project='quantum-sim')
from qiskit import Aer
import copy
from qiskit import quantum_info as qi
from qiskit import execute
import numpy as np

class Quantum_Device:
    
    def __init__(self, backend = 'ibm_lagos', layout = [0,1,3,5]):
        self.backend = provider.get_backend(backend)
        self.layout = layout


    def get_results(self,circs_in, method = 'matrix',save_id_file = None):
        circs = copy.deepcopy(circs_in)
        #Exact Calculation
        if method == 'matrix':
            # add results
            r = []
            for l in range(len(circs)):
                circs[l].remove_final_measurements()
                wave = qi.Statevector.from_instruction(circs[l])
                r_l = wave.probabilities_dict()
                r.append(r_l)
        #Qiskit Simulation        
        if method == "simulator":
            sim = Aer.get_backend("qasm_simulator")
            r = execute(circs, backend = sim).result().get_counts() 
        #Device
        if method == "quantum":
            backend = self.backend
            layout = self.layout
            job = backend.run(transpile(circs, backend,initial_layout = layout), meas_level=2, shots=8192)
            job_id = job.job_id()
            print(job_id)
            np.save(save_id_file,[job_id])
            r = job.result().get_counts()
        return r
    
    def normalize_results(self,results):
        results_out = []
        for result in results:
            result_out = {}
            states = list(result.keys())
            total = 0
            for state in states:
                total += result[state]
            for state in states:
                result_out[state] = result[state]/total
            results_out.append(result_out)
        return results_out
    
    def chop_results(self,results):
        results_chop = []
        for result in results:
            result_chop = {}
            states = list(result.keys())
            for state in states:
                if result[state] > 10**(-5):
                    result_chop[state] = result[state]
            results_chop.append(result_chop)
        return results_chop
