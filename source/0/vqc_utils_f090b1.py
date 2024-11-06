# https://github.com/ezeeEric/qicl/blob/1d87f6a077f382bce964e0f88bd331d29e7e4981/vqcore/vqc_utils.py
import numpy as np
from qiskit.aqua import QuantumInstance
from utils.tools import timelogDict,chronomat


@chronomat
def getSimulationInstance(nShots=1000):
    from qiskit import BasicAer
    backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(
            backend = backend,
            shots=nShots)
    return quantum_instance  

@chronomat
def getIBMQInstance(nShots=1000):
    from qiskit import IBMQ
    provider = IBMQ.load_account()
    from qiskit.providers.ibmq import least_busy
    least_busy_device = least_busy(provider.backends(simulator=False))
    print("Using currently least busy device: ", least_busy_device)
    quantum_instance = QuantumInstance(
            backend = least_busy_device,
            shots=nShots)
    return quantum_instance

@chronomat
def trainVQC(vqc,qi):
    result=None
    print("Running VQC")
    result = vqc.run(qi)
    print("testing success ratio: ", result['testing_accuracy'])
    return result

@chronomat
def predictVQC(vqc,testData):
    predicted_probs, predicted_labels = vqc.predict(testData)
    return predicted_probs, predicted_labels 

##############from qiskit.vqc
def assign_label(measured_key, num_classes):
    """
    Classes = 2:
    - If odd number of qubits we use majority vote
    - If even number of qubits we use parity
    Classes = 3
    - We use part-parity
        {ex. for 2 qubits: [00], [01,10], [11] would be the three labels}
    Args:
        measured_key (str): measured key
        num_classes (int): number of classes
    """
    measured_key = np.asarray([int(k) for k in list(measured_key)])
    num_qubits = len(measured_key)
    if num_classes == 2:
        if num_qubits % 2 != 0:
            total = np.sum(measured_key)
            return 1 if total > num_qubits / 2 else 0
        else:
            hamming_weight = np.sum(measured_key)
            is_odd_parity = hamming_weight % 2
            return is_odd_parity

    elif num_classes == 3:
        first_half = int(np.floor(num_qubits / 2))
        modulo = num_qubits % 2
        # First half of key
        hamming_weight_1 = np.sum(measured_key[0:first_half + modulo])
        # Second half of key
        hamming_weight_2 = np.sum(measured_key[first_half + modulo:])
        is_odd_parity_1 = hamming_weight_1 % 2
        is_odd_parity_2 = hamming_weight_2 % 2

        return is_odd_parity_1 + is_odd_parity_2

    else:
        total_size = 2**num_qubits
        class_step = np.floor(total_size / num_classes)

        decimal_value = measured_key.dot(1 << np.arange(measured_key.shape[-1] - 1, -1, -1))
        key_order = int(decimal_value / class_step)
        return key_order if key_order < num_classes else num_classes - 1


def cost_estimate(probs, gt_labels, shots=None):
    """Calculate cross entropy
    # shots is kept since it may be needed in future.
    Args:
        shots (int): the number of shots used in quantum computing
        probs (numpy.ndarray): NxK array, N is the number of data and K is the number of class
        gt_labels (numpy.ndarray): Nx1 array
    Returns:
        float: cross entropy loss between estimated probs and gt_labels
    """
    mylabels = np.zeros(probs.shape)
    for i in range(gt_labels.shape[0]):
        whichindex = gt_labels[i]
        mylabels[i][whichindex] = 1

    def cross_entropy(predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        tmp = np.sum(targets*np.log(predictions), axis=1)
        ce = -np.sum(tmp)/N
        return ce

    x = cross_entropy(probs, mylabels)
    return x


def cost_estimate_sigmoid(shots, probs, gt_labels):
    """Calculate sigmoid cross entropy

    Args:
        shots (int): the number of shots used in quantum computing
        probs (numpy.ndarray): NxK array, N is the number of data and K is the number of class
        gt_labels (numpy.ndarray): Nx1 array
    Returns:
        float: sigmoid cross entropy loss between estimated probs and gt_labels
    """
    # Error in the order of parameters corrected below - 19 Dec 2018
    # x = cost_estimate(shots, probs, gt_labels)
    x = cost_estimate(probs, gt_labels, shots)
    loss = (1.) / (1. + np.exp(-x))
    return loss


def return_probabilities(counts, num_classes):
    """Return the probabilities of given measured counts
    Args:
        counts ([dict]): N data and each with a dict recording the counts
        num_classes (int): number of classes
    Returns:
        numpy.ndarray: NxK array
    """

    probs = np.zeros(((len(counts), num_classes)))
    for idx in range(len(counts)):
        count = counts[idx]
        shots = sum(count.values())
        for k, v in count.items():
            label = assign_label(k, num_classes)
            probs[idx][label] += v / shots
    return probs
