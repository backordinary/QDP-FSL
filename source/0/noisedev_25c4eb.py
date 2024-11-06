# https://github.com/quantum-italy-developer/qiskit_app_carbon_design/blob/ef83f56fcf9af8437a34044642fc110068920c05/noisedev.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

#IBMQ.enable_account(token, url='https://quantumexperience.ng.bluemix.net/api') 
#IBMQ.active_accounts()
#IBMQ.backends()

def setup_noise(circuit,shots):
    """
    This fucntion run a circuit on a simulated ibmq_16_melbourne device with noise
    """
    # Basic device noise model
    # List of gate times
    # Note that the None parameter for u1, u2, u3 is because gate
    # times are the same for all qubits
    gate_times = [
        ('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
        ('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
        ('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
        ('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
        ('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
        ('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
        ('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)
    ]

    device = IBMQ.get_backend('ibmq_16_melbourne')
    properties = device.properties()
    coupling_map = device.configuration().coupling_map

    # Construct the noise model from backend properties
    # and custom gate times
    noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)

    # Get the basis gates for the noise model
    basis_gates = noise_model.basis_gates

    # Select the QasmSimulator from the Aer provider
    simulator = Aer.get_backend('qasm_simulator')

    # Execute noisy simulation and get counts
    result_noise = execute(circuit, simulator,shots=shots,noise_model=noise_model, 
                    coupling_map=coupling_map, basis_gates=basis_gates).result().get_counts(circuit)
#    print(result_noise)
    #counts_noise = result_noise.get_counts(circ)
    return result_noise 

def emo_noise():
    """
   This function simulate the emoticon function with noise, to perform noise we are limited to use
   just a streactly set of emoticon because of constraints coming from the real device.
    """
    # set up registers and program
    qr = QuantumRegister(14)
    cr = ClassicalRegister(14)
    qc = QuantumCircuit(qr, cr)
    # rightmost seven (qu)bits have ')' = 0101001
    qc.x(qr[0])
    qc.x(qr[3])
    qc.x(qr[5])
    # second seven (qu)bits have superposition of
    # '8' = 0111000
    # ';' = 0111011
    # these differ only on the rightmost two bits
    qc.h(qr[8]) # create superposition on 9
    qc.cx(qr[8],qr[7]) # spread it to 8 with a CNOT
    qc.x(qr[10])
    qc.x(qr[11])
    qc.x(qr[12])
    # measure
    for j in range(14):
        qc.measure(qr[j], cr[j])
    shots=400
    ## RUN THE SIMULATION 
    return synt_chr(setup_noise(qc,shots))


def synt_chr(stats, shots=None):
    """
    This function transforms the binary the string of bits into ascii character 
    """
    n_list=[]
    for bitString in stats:
        char = chr(int( bitString[0:7] ,2)) # get string of the leftmost 7 bits and convert to an ASCII character
        char += chr(int( bitString[7:14] ,2)) # do the same for string of rightmost 7 bits, and add it to the previous character
        #prob = stats[bitString] / shots # fraction of shots for which this result occurred
        n_list.append({"shots":stats[bitString],"value":char})
    return n_list

def setup_noise_s(circuit,shots):
    """
    This fucntion run a circuit on a simulated ibmq_16_melbourne device with noise
    """
    # Basic device noise model
    # List of gate times
    # Note that the None parameter for u1, u2, u3 is because gate
    # times are the same for all qubits
    gate_times = [
        ('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
        ('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
        ('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
        ('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
        ('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
        ('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
        ('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)
    ]

    device = IBMQ.get_backend('ibmq_16_melbourne')
    properties = device.properties()
    coupling_map = device.configuration().coupling_map

    # Construct the noise model from backend properties
    # and custom gate times
    noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)

    # Get the basis gates for the noise model
    basis_gates = noise_model.basis_gates

    # Select the QasmSimulator from the Aer provider
    simulator = Aer.get_backend('qasm_simulator')

    # Execute noisy simulation and get counts
    result_noise = execute(circuit, simulator,shots=shots,noise_model=noise_model, 
                    coupling_map=coupling_map, basis_gates=basis_gates).result()
#    print(result_noise)
    #counts_noise = result_noise.get_counts(circ)
    return result_noise 