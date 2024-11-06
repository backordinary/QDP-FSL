# https://github.com/ChitambarLab/QiskitSummerJam2020/blob/cff5abb69997759f043dd1ca690735718c5c66bc/device_independent_test/entanglement.py
import numpy as np
from qiskit import QuantumCircuit

# @brief    Circuit creating the + bell state
# @returns  Two qubit circuit resulting in the + bell state
def create_bell_state():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0,1)
    return circuit

# @brief    Computes the CHSH expecation value from a job
# @params   counts: dictionary of measurements
#           shots:  number of shots
# @returns  float expectation value
def compute_expectation_for_CHSH(counts, shots):
    return (counts["00"] + counts["11"] - counts["01"] - counts["10"])/shots

# @brief    Runs a CHSH test (seperate circuits)
# @params   dispatcher: QuantumDispatcher to run circuits and transmit states
#           tolerance: max deviation from quantum expectation value allowed
#           shots: number of shots to run
# @returns  Tuple of (pass/fail, test value)
# @note     measure_all() cannot be used in the construction of the circuits below
#               as it will add another classical register
def run_test(dispatcher, tolerance=0.4, shots=1000):
    pre_ops = [create_bell_state()]

    # Alice's measurements (Z or X basis)
    alice_z = alice_Z()
    alice_z.measure(0,0)
    alice_x = alice_X()
    alice_x.measure(0,0)

    # Bob's measurements (W or V basis)
    bob_w = bob_W()
    bob_w.measure(1,1)
    bob_v = bob_V()
    bob_v.measure(1,1)

    post_ops = [[alice_z,alice_x],[bob_w,bob_v]]

    # run all permutations through the dispatcher
    counts = dispatcher.batch_run_and_transmit(pre_ops, post_ops, shots)
    expected_ZW = compute_expectation_for_CHSH(counts[0], shots)
    expected_ZV = compute_expectation_for_CHSH(counts[1], shots)
    expected_XW = compute_expectation_for_CHSH(counts[2], shots)
    expected_XV = compute_expectation_for_CHSH(counts[3], shots)

    test_val = (expected_ZW + expected_ZV + expected_XW - expected_XV)
    expectation_value = 2.82842712475 # quantum expectation value

    return (abs(test_val-expectation_value) <= tolerance, test_val)

# @brief    Runs as CHSH test by paralleling into two jobs on 4 qubits
#               Each is a specific measurement by alice and both W and V by Bob
# @params   dispatcher: QuantumDispatcher to run circuits and transmit states
#           tolerance: max deviation from quantum expectation value allowed
#           shots: number of shots to run
# @returns  Tuple of (pass/fail, test value)
# @note     Measurements are added to individual circuits to be then
#               dispatched to respective devices by the dispatcher
def run_test_parallel(dispatcher,tolerance=0.4,shots=1000):
    # create two bell states across 4 registers
    pre_qc = QuantumCircuit(4)
    pre_qc.append(create_bell_state(),[0,1])
    pre_qc.append(create_bell_state(),[2,3])

    # create measurement circuits
    qc_z = QuantumCircuit(4,4) # alice measuring in Z basis
    qc_z.append(alice_Z(),[0,1])
    qc_z.append(alice_Z(),[2,3])
    qc_z.measure(0,0)
    qc_z.measure(2,2)

    qc_x = QuantumCircuit(4,4) # alice measuring in X basis
    qc_x.append(alice_X(),[0,1])
    qc_x.append(alice_X(),[2,3])
    qc_x.measure(0,0)
    qc_x.measure(2,2)

    # one circuit for bob measuring in W and V
    qc_wv = QuantumCircuit(4,4) # bob measuring in w basis
    qc_wv.append(bob_W(),[0,1])
    qc_wv.append(bob_V(),[2,3])
    qc_wv.measure(1,1)
    qc_wv.measure(3,3)

    # run all combinations
    pre_ops = [pre_qc]
    post_ops = [[qc_z,qc_x],[qc_wv]]
    test_val = parse_parallel_data(
        dispatcher.batch_run_and_transmit(pre_ops,post_ops,shots),
        shots
    )

    expectation_value = 2.82842712475 # quantum expectation value

    return (abs(test_val-expectation_value) <= tolerance, test_val)

# @brief    Parses the data from running 2 cases at once on 4 registers
# @params   counts: dictionary of counts from job
# @returns  The expectation value of the CHSH test
# @note     The inputted keys in counts have to be reversed as the last
#               character corresponds to the first qubit
def parse_parallel_data(counts, shots):
    counts_zw = { "00":0, "01":0, "10":0, "11":0 }
    counts_zv = { "00":0, "01":0, "10":0, "11":0 }
    for key in counts[0]:
        # reverse the key
        key_rev = key[::-1]
        counts_zw[key_rev[0:2]] += counts[0][key]
        counts_zv[key_rev[2:4]] += counts[0][key]

    counts_xw = { "00":0, "01":0, "10":0, "11":0 }
    counts_xv = { "00":0, "01":0, "10":0, "11":0 }
    for key in counts[0]:
        # reverse the key
        key_rev = key[::-1]
        counts_xw[key_rev[0:2]] += counts[1][key]
        counts_xv[key_rev[2:4]] += counts[1][key]

    expected_ZW = compute_expectation_for_CHSH(counts_zw, shots)
    expected_ZV = compute_expectation_for_CHSH(counts_zv, shots)
    expected_XW = compute_expectation_for_CHSH(counts_xw, shots)
    expected_XV = compute_expectation_for_CHSH(counts_xv, shots)

    return expected_ZW + expected_ZV + expected_XW - expected_XV

# @brief    A 2 qubit circuit with no operations (Z measurement)
def alice_Z():
    qc = QuantumCircuit(2,2)
    return qc

# @brief    A 2 qubit circuit for measuring the first in the X basis
def alice_X():
    qc = QuantumCircuit(2,2)
    qc.h(0)
    return qc

# @brief    A 2 qubit circuit for measuring the second in the W basis
def bob_W():
    qc = QuantumCircuit(2,2)
    qc.s(1)
    qc.h(1)
    qc.t(1)
    qc.h(1)
    return qc

# @brief    A 2 qubit circuit for measuring the second in the V basis
def bob_V():
    qc = QuantumCircuit(2,2)
    qc.s(1)
    qc.h(1)
    qc.tdg(1)
    qc.h(1)
    return qc
