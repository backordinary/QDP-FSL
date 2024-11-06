# https://github.com/austinjhunt/shors/blob/c5cf4d9484f3b6c459f6a770db745e1f4edc0d5e/src/qskt/qft.py

from qiskit import QuantumCircuit
from math import pi
from base import Base
from qiskit.circuit.library import QFT

class QuantumFourierTransform(Base):

    def __init__(self, name: str = 'QFT', verbose: bool = False):
        super().__init__(name, verbose=verbose)
    

    def qft_dagger(self, n):
        """Apply N-qubit QFTdagger to the first n qubits of a circuit.
        
        QFT dagger is the conjugate transpose of the Quantum Fourier Transform,
        where the quantum fourier transform is used to find a frequency of a 
        given superposition. 
        """ 
        # Defining a new quantum circuit of n qubits. 
        circuit = QuantumCircuit(n)

        # for each qubit in the first half of that circuit
        for qubit in range(n//2):
            # Apply a SWAP gate to swap:
            #  qubit 0 with the n - 1 qubit
            #  qubit 1 with the n - 2 qubit
            #  qubit 2 with the n - 3 qubit
            # ... such that you are inverting the first n qubits
            circuit.swap(qubit, n - qubit - 1)

        # for each qubit j in the full circuit
        for j in range(n):
            # for each qubit preceding j 
            for m in range(j): 
                # apply a Controlled-Phase gate
                # This is a diagonal and symmetric gate that induces a 
                # phase/rotation on the state of the target qubit, depending on the control state.
                # Define the rotation angle as (-pi) / 2^(j-m)
                rotation_angle = -pi/float(2**(j-m))
                # apply the controlled-phase gate to qubit j using that rotation angle, 
                # using m as the control qubit, and using j as the target qubit
                circuit.cp(rotation_angle, control_qubit=m, target_qubit=j)
            
            # Apply Hadamard gate to qubit j to 
            # put it into a superposition such that probability of 0 = probability of 1
            circuit.h(j)

        # Give the Quantum Fourier Transform Dagger (conjugate transpose) 
        # circuit a name and then return it 
        circuit.name = "QFT†"
        return circuit

    def apply_qft(self, num_qubits: int = 1, use_qiskit_lib: bool = False ):
        """ This circuit is explained very clearly by Abraham Asfaw in his 
        lecture at https://www.youtube.com/watch?v=pq2jkfJlLmY.
        
        """
        if use_qiskit_lib:
            return QFT(num_qubits=num_qubits)
        circuit = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            # For each qubit x_1, x_2, ... x_num_qubits 
            # apply a hadamard to the qubit
            circuit.h(qubit)
            for other_qubit in range(qubit+1, num_qubits):
                # then apply a series of phases/rotations
                # using controlled phase / unitary rotation / UROT_k
                # gate that will apply phase conditionally
                # depending on control qubit.
                
                # apply UROT_k (unitary rotation)
                circuit.cu1(
                    pi / (2 ** (other_qubit - qubit)), # phase to apply 
                    other_qubit, # control qubit
                    qubit # target qubit 
                )
        return circuit
    
if __name__ == "__main__": 
    qft = QuantumFourierTransform()
    qft.apply_qft(num_qubits=4).draw(filename='qft-4-qubits.jpg')