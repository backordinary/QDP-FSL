# https://github.com/quantcert/quantcert.github.io/blob/369089e3319ead9947abb3694f3ad64dac255501/Mermin-hypergraph-states/Qiskit/app/mermin_on_qiskit/evaluation.py
import numpy as np
from qiskit import *

from . import run
from . import basis_change


def mermin_IBM(n):
    """ Returns the Mermin polynomials under a vector form. This form helps to 
        form the corrects monomials that involves in every mermin evaluation.

    Example :
        In this case, the involving monomials are only the second one, the third 
        one, the fith one and the last one because the others are equal to zero.
        >>> mermin_IBM(3)
        [0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, -0.5]

    :param int n: The number of qubits.
    :returns: list(float) -- The list of numbers corresponding to the existence 
        and the value each monomial.
    """
    if n == 1:
        # M1 is equal to |0>
        mn = [1, 0]
    else:
        # Recursion formula
        mn = 0.5 * (np.kron(mermin_IBM(n - 1), [1, 1]) + np.kron(mermin_IBM(n - 1)[::-1], [1, -1]))
    return list(mn)


def measures_exploitation(measures_dictionary, shots):
    """ Calculates the measurements probabilities

    For every possible cases (for example, with n = 2 : 00 01 10 11), the 
    probability to get this combination when measuring is calculated.

    In order to obtain this probabilities, we sum the values of the cases where 
    the number of 1 in the measurement is even and when it's then odd.
    
    Example :
        even_results = values of 00 and 11 measurements
        odd_results = values of 01 and 10 measurements

    :param dict measures_dictionary: the dictionary containing the measurements 
        and their values.
    :param int shots: the number of times that the measurements are made. This 
        is only in case of a local test.
    :returns: float -- The total probability of the dictionary measurement.
    """
    even_results = 0
    odd_results = 0
    for measure in measures_dictionary:
        number_of_1 = measure.count('1')
        if number_of_1 % 2 == 0:
            even_results += measures_dictionary[measure]
        else:
            odd_results += measures_dictionary[measure]
    return (even_results - odd_results) / shots


def evaluate_monomial(
        n, n_measure, circuit, a_a_p_coeffs, shots, 
        is_simulation=True, monitor=False, local=True):
    """ Draws the circuit if there are some additions and runs it to get the 
        measurement of a monomial

    :param int n: The number of qubits.
    :param int n_measure: The measurement to be performed. Dictates whether a_i 
        or a'_i is used on each wire.
    :param QuantumCircuit circuit: The original quantum circuit.
    :param array[float] a_a_p_coeffs: The coefficients of the matrices used to 
        calculate Mermin operators.
    :param int shots: The number of repetitions of each circuit. Default: 1024.
    :param boolean is_simulation: This determines if we are in a case of a local
        test or a real IBM machine test.
    :param boolean monitor: If true a monitor is attached to the job.
    :param boolean local: If true, the job run on a local simulator.
    :returns: float -- The result of the measurement probabilities on one 
        monomial.
    """
    circuit_size = circuit.num_qubits
    circuit_aux = QuantumCircuit(circuit_size, n)

    basis_change.U3_gates_placement(n, n_measure, a_a_p_coeffs, circuit_aux)
    circuit_mesure = circuit + circuit_aux

    result = run.runCircuit(
        circuit_mesure, shots=shots, 
        simulation=is_simulation, monitor=monitor, local=local)

    measure_monomial = measures_exploitation(result, shots)

    return measure_monomial


def evaluate_polynomial(
        n, circuit, a_a_p_coeffs, shots=1024, 
        is_simulation=True, monitor=False, local=True):
    """ Makes all the implementation and calculation

    Caution!: 
        The IBMQ account must be loaded before the execution of this 
        function if the variable is_simulation is set to False.

    :param int n: The number of qubits.
    :param QuantumCircuit circuit: The original quantum circuit.
    :param list[list[any]] a_a_p_coeffs: Lists of lists of elements as described 
        above (packed coefficients).
    :param int shots: The number of times that the measurements are made. This 
        is only in case of a local test.
    :param boolean is_simulation: To specify if the codes are to run locally or 
        on the IBM machine.
    :param boolean monitor: If true a monitor is attached to the job.
    :param boolean local: If true, the job run on a local simulator.
    :return: float -- The result of all the calculations.
    """
    mermin_polynomial = mermin_IBM(n)
    total_result = 0
    for i in range(2 ** n):
        if mermin_polynomial[i] != 0:
            measure_monomial = evaluate_monomial(
                n, i, circuit, a_a_p_coeffs, shots, is_simulation, 
                monitor=monitor, local=local)
            total_result += mermin_polynomial[i] * measure_monomial
    return abs(total_result)
