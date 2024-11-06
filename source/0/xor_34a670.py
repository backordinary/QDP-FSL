# https://github.com/AlexPoiron/QUML/blob/183b3ef16346c9aa63307fe08e97a2e3cdc02e80/XOR.py
import numpy as np
import pandas as pd
import qiskit
from proglearn.sims import generate_gaussian_parity
from problem import Problem, rescaleFeature


NSHOTS = 10000
qs = qiskit.Aer.get_backend('qasm_simulator')
#Token used for the IBMQ circuits
TOKEN = "73547946bd0f7f1e1b48368ac35872c76b8bd0100e1e84ea0411076c44208af1127b3b69f345e138c07b03c36809afba05d2e5d9aa1eac3e4d352be42575af06"

class XOR(Problem):
    """XOR class corresponding to the 2nd problem in the paper.

    Args:
        Problem (class): The super class
    """
    def __init__(self):
        super().__init__()
        self.name = "XOR"
        self.theta_init = np.random.uniform(0, 2*np.pi, 4)
    
    def get_dict(self):
        """Get the dictionnary corresponding to the problem
        
        """
        return {
        "1" : ["10","01"],
        "0" : ["00","11"]
        }
    
    def get_dicinv_XOR(self):
        """Get the inverted dictionnary from the original dictionnary. We have here a specification for the XOR one

        Returns:
            the inverted dictionnary
        """
        dict = self.get_dict()
        dicinv = {}
        for k in dict:
            for i in dict[k]:
                dicinv.update({i : k})
        return dicinv

    def build_circuit(self, theta, omega):
        """Build the quantum circuit corresponding to the problem

        Args:
            theta (np.ndarray): the optimized parameter found in the training
            omega (pd.Series): row on the test_set

        Returns:
            Return the qunatum circuit built.
        """
        qc = qiskit.QuantumCircuit(2)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(omega[0], 0)
        qc.rz(omega[1], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[0], 0)
        qc.rz(theta[1], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[2], 0)
        qc.rz(theta[3], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        
        self.circuit = qc
        return qc
    
    def prediction_dict(self, theta, omega):
        """Get the measurement of our quantum circuit. This measurement gives a count on each possible output possible

        Args:
            theta (np.ndarray): the optimized parameter obtained with the training
            omega (pd.Series): row on the test set

        Returns:
            A qiskit object that is auite similar to a dictionnary with counts on each output qbits.
        """
        qc = qiskit.QuantumCircuit(2, 2)
        qc.append(self.build_circuit(theta, omega), range(2))
        qc.measure(range(2), range(2))
        
        job = qiskit.execute(qc, shots=NSHOTS, backend=qs)
        res = {'10':0, '01':0,'00':0, '11':0}
        c = job.result().get_counts()
        for key in c:
            res[key] = c[key]
        return res
    
    def prediction_dict_IBMQ(self, theta: np.ndarray, omega: pd.Series) -> qiskit.result.counts.Counts:
        """Get the measurement of our quantum circuit. This measurement gives a count on each possible output possible. This, time
           we use online quantum material.

        Args:
            theta (np.ndarray): the optimized parameter obtained with the training
            omega (pd.Series): row on the test set

        Returns:
            A qiskit object that is auite similar to a dictionnary with counts on each output qbits.
        """
        qiskit.IBMQ.save_account(TOKEN, overwrite=True) 
        provider = qiskit.IBMQ.load_account()
        backend = qiskit.providers.ibmq.least_busy(provider.backends())

        qc = qiskit.QuantumCircuit(2, 2)
        qc.append(self.build_circuit(theta, omega), range(2))
        qc.measure(range(2), range(2))

        mapped_circuit = qiskit.transpile(qc, backend=backend)
        qobj = qiskit.assemble(mapped_circuit, backend=backend, shots=NSHOTS)

        job = backend.run(qobj)
        print(job.status())
        res = job.result().get_counts()

        return res
    
    def get_df(self):
        """Create a Pandas Dataframe

        Returns:
            the Dataframe
        """
        X_rxor, y_rxor = generate_gaussian_parity(1000, angle_params=np.pi / 4)
        X_rxor0 = [values[0] for values in X_rxor]
        X_rxor1 = [values[1] for values in X_rxor]
        list_dict = {'Values0' : X_rxor0,
                     'Values1' : X_rxor1,
                     'class' : [str(value) for value in y_rxor]} 
        df = pd.DataFrame(list_dict)
        attributes = df.columns[:-1]
        for x in attributes:
            df[x] = rescaleFeature(df[x])
        return df