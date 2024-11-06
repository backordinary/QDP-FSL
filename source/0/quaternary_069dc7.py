# https://github.com/AlexPoiron/QUML/blob/183b3ef16346c9aa63307fe08e97a2e3cdc02e80/Quaternary.py
import numpy as np
import pandas as pd
import qiskit
from problem import Problem, rescaleFeature
from sklearn.datasets import make_classification

NSHOTS = 1500
qs = qiskit.Aer.get_backend('qasm_simulator')
#Token used for the IBMQ circuits
TOKEN = "73547946bd0f7f1e1b48368ac35872c76b8bd0100e1e84ea0411076c44208af1127b3b69f345e138c07b03c36809afba05d2e5d9aa1eac3e4d352be42575af06"

class Quaternary(Problem):
    """Quaternary class corresponding to the 4th problem in the paper.

    Args:
        Problem (class): The super class
    """
    def __init__(self):
        super().__init__()
        self.name = "Quaternary"
        self.theta_init = np.random.uniform(0, 2*np.pi, 12)

    def get_dict(self):
        """Get the dictionnary corresponding to the problem
        
        """
        return {
            "0" : "00",
            "1" : "01",
            "2" : "10",
            "3" : "11"
            }
       
    def build_circuit(self, theta, omega):
        """Build the quantum circuit corresponding to the problem

        Args:
            theta (np.ndarray): the optimized parameter found in the training
            omega (pd.Series): row on the test_set

        Returns:
            Return the qunatum circuit built.
        """
        qc = qiskit.QuantumCircuit(2)
        if 0 : qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(omega[(2*0) % 4], 0)
        qc.rz(omega[(2*0+1) % 4], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[2*0], 0)
        qc.rz(theta[2*0+1], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        if 0 == 1 or 0 == 3:
            qc.cz(0, 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            qc.rz(theta[2*0+2], 0)
            qc.rz(theta[2*0+3], 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            0+=1
            
        if 1 : qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(omega[(2*1) % 4], 0)
        qc.rz(omega[(2*1+1) % 4], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[2*1], 0)
        qc.rz(theta[2*1+1], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        if 1 == 1 or 1 == 3:
            qc.cz(0, 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            qc.rz(theta[2*1+2], 0)
            qc.rz(theta[2*1+3], 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            1+=1
            
        if 2 : qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(omega[(2*2) % 4], 0)
        qc.rz(omega[(2*2+1) % 4], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[2*2], 0)
        qc.rz(theta[2*2+1], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        if 2 == 1 or 2 == 3:
            qc.cz(0, 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            qc.rz(theta[2*2+2], 0)
            qc.rz(theta[2*2+3], 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            2+=1
            
        if 3 : qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(omega[(2*3) % 4], 0)
        qc.rz(omega[(2*3+1) % 4], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[2*3], 0)
        qc.rz(theta[2*3+1], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        if 3 == 1 or 3 == 3:
            qc.cz(0, 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            qc.rz(theta[2*3+2], 0)
            qc.rz(theta[2*3+3], 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            3+=1
            
        if 4 : qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(omega[(2*4) % 4], 0)
        qc.rz(omega[(2*4+1) % 4], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[2*4], 0)
        qc.rz(theta[2*4+1], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        if 4 == 1 or 4 == 3:
            qc.cz(0, 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            qc.rz(theta[2*4+2], 0)
            qc.rz(theta[2*4+3], 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            4+=1
            
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
        X, y = make_classification(
            n_samples=5000,  
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_classes=4,
        )
        df = pd.DataFrame(X)
        df['class'] = y
        df['class'] = df['class'].astype(str)
        attributes = df.columns[:-1]
        for x in attributes:
            df[x] = rescaleFeature(df[x])
        return df