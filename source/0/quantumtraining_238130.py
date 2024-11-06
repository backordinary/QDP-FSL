# https://github.com/Red2Je/IBMInternship/blob/3bd7034c1a4245c134b44c682c549491dfed3ce6/WSL%20works/torch%20gpu/QuantumTraining.py
#La description de cette classe est disponible dans le fichier Hybrid neural network.ipynb
import qiskit
from qiskit import transpile, assemble
import numpy as np

class QuantumCircuitBuilder:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, backend, shots,device):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        
        self._circuit.measure_all()
        # ---------------------------
        self.device = device
        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds = [{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        expectation_list = []
        if not isinstance(result,dict):
            for res in result:
                counts = np.array(list(res.values()))
                states = np.array(list(res.keys())).astype(float)


                # Compute probabilities for each state
                probabilities = counts / self.shots
                # Get state expectation
                expectation_list.append(np.sum(states * probabilities))
            return np.array(expectation_list)
        else:
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probabilities = counts/self.shots
            expectation = np.sum(states*probabilities)
            return np.array([expectation])