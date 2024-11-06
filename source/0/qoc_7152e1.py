# https://github.com/shukob/qocc/blob/f07577c8c27fc53611e8533a2d7f9db192158562/QOC/qoc.py
# -----------------------------------------------------------------------------
# Distributed under the GNU General Public License.
#
# Contributors: Nicolas Melo (nmo@cin.ufpe.br)
#               Rodrigo Sousa (--)
#               Adenilton Silva ajsilva@cin.ufpe.br
# -----------------------------------------------------------------------------
# References:
#
# ESANN2020 paper
#
# Schuld, M., Fingerhuth, M., & Petruccione, F. (2017). Implementing a distance
# -based classifier with a quantum interference circuit. EPL (Europhysics 
# Letters), 119(6), 60002.
# -----------------------------------------------------------------------------
# File description:
#
# Quantum one-class classifier implementation on qiskit
# -----------------------------------------------------------------------------
# Notes:
#
# This classifier is based on the classifier presented in 
# https://arxiv.org/abs/1703.10793
# -----------------------------------------------------------------------------

from qiskit import (QuantumCircuit, ClassicalRegister, QuantumRegister, 
                    execute, BasicAer, IBMQ)
from qiskit.tools.monitor import job_monitor

class QOC:
    
    def initializeRegisters(self):
        """
        Creates quantum and classical registers
        """
        
        self.qr = QuantumRegister(3)
        self.cr = ClassicalRegister(1)
        
        # Name the individual qubits in quantum register
        self.ancilla_qubit = self.qr[0]
        self.index_qubit = self.qr[1]
        self.data_qubit = self.qr[2]
        
    def createCircuit(self, test, training):
        """
        Creates the quantum circuit loading vector angles 'test' and 'training'
        """
        
        # Create empty quantum circuit
        self.qc = QuantumCircuit(self.qr, self.cr)
        
        # Put the ancilla and the index qubits into uniform superposition
        self.qc.h(self.ancilla_qubit)
        self.qc.h(self.index_qubit)
        self.qc.barrier()
        
        # Load the vector to be classified
        self.load_test_vector(test)
        
        # Flip the ancilla qubit moves the input vector to the |0> 
        # state of the ancilla
        self.qc.x(self.ancilla_qubit)
        self.qc.barrier()
        
        # Load the first training vector
        self.load_training_vector(training[0])
        
        # Flip the index qubit moves the first training vector to the |0> 
        # state of the index qubit
        self.qc.x(self.index_qubit)
        self.qc.barrier()
        
        # Load the second training vector
        self.load_training_vector(training[1])
        
        # The Hadamard gate interferes the copies of the test vector with the
        # training vectors
        self.qc.h(self.ancilla_qubit)
        self.qc.barrier()
        
        # Measure the ancilla qubit and store the result in the classical
        # register
        self.qc.measure(self.ancilla_qubit, self.cr)
        
        return self.qc
    
    def load_test_vector(self, theta):
        """
        Loads the test vector from its theta angle
        """
        
        self.qc.cx(self.ancilla_qubit, self.data_qubit)
        self.qc.ry(theta*-1, self.data_qubit) 
        self.qc.cx(self.ancilla_qubit, self.data_qubit) 
        self.qc.ry(theta, self.data_qubit)
        
        self.qc.barrier()
        
    def load_training_vector(self, theta):
        """
        Loads the training vector from its theta angle
        """
        
        self.qc.ccx(self.ancilla_qubit, self.index_qubit, self.data_qubit)
        self.qc.cx(self.index_qubit, self.data_qubit)
        self.qc.ry(theta, self.data_qubit)
        self.qc.cx(self.index_qubit, self.data_qubit)
        self.qc.ry(theta*-1, self.data_qubit)
        
        self.qc.ccx(self.ancilla_qubit, self.index_qubit, self.data_qubit)
        self.qc.cx(self.index_qubit, self.data_qubit)
        self.qc.ry(theta*-1, self.data_qubit)
        self.qc.cx(self.index_qubit, self.data_qubit)
        self.qc.ry(theta, self.data_qubit)
        
        self.qc.barrier()
        
    def runSim(self, quantum_circuit):
        """
        Compiles and runs the quantum circuit on a simulator backend
        """
        
        backend_sim = BasicAer.get_backend('qasm_simulator')
        job_sim = execute(quantum_circuit, backend_sim, shots=1024)

        return job_sim.result()
    
    def runReal(self, quantum_circuit):
        """
        Compiles and runs the quantum circuit on a real quantum processor
        """
        
        # Update/save/load IBM account credentials
        #IBMQ.update_account()
        #IBMQ.save_account('MY_API_TOKEN')
        IBMQ.load_account()
        
        provider = IBMQ.get_provider(group='open')
        backend_real = provider.get_backend('ibmq_essex')
        job = execute(quantum_circuit, backend_real, shots=1024)
        
        # Monitor the status of a IBMQ job instance
        job_monitor(job)
        
        return job.result()
    
    def classifySim(self, test_vector, training_set):
        """
        Classifies the test vector with the quantum one-class classifier 
        using simulator
        """
        
        self.initializeRegisters()
        qc = self.createCircuit(test_vector, training_set)

        result = self.runSim(qc)
        result_counts = result.get_counts(qc)
        total_samples = sum(result_counts.values())

        # Define lambda function that retrieves only results where the ancilla 
        # is in the |0> state
        ancilla_func = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '0']
        
        # Retrive counting values for the ancilla qubit by applying ancilla_func
        ancilla_counts = dict(ancilla_func(result_counts))
        ancilla_samples = sum(ancilla_counts.values())
        
        # Probability of ancilla qubit is in the |0> state
        ancilla_prob = ancilla_samples/total_samples
        
        return total_samples, ancilla_counts, ancilla_samples, ancilla_prob
        
    def classifyReal(self, test_vector, training_set):
        """
        Classifies the test vector with the quantum one-class classifier 
        using a real quantum processor
        """
        
        self.initializeRegisters()
        qc = self.createCircuit(test_vector, training_set)

        realExec = self.runReal(qc)
        result_counts = realExec.get_counts(qc)
        total_samples = sum(result_counts.values())

        # Define lambda function that retrieves only results where the ancilla 
        # is in the |0> state
        ancilla_func = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '0']
        
        # Retrive counting values for the ancilla qubit by applying ancilla_func
        ancilla_counts = dict(ancilla_func(result_counts))
        ancilla_samples = sum(ancilla_counts.values())
        
        # Probability of ancilla qubit is in the |0> state
        ancilla_prob = ancilla_samples/total_samples
        
        return total_samples, ancilla_counts, ancilla_samples, ancilla_prob