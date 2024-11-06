# https://github.com/ogorodnikov/m1/blob/922d775cfbbe0dbfd68470aae6209a4c2a6d3ddc/app/core-service/core/algorithms/iqae.py
from math import pi, sin, asin, acos, log, log2, sqrt

from scipy.stats import beta

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister

from qiskit import Aer

from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance


### Service functions

def chernoff_confidence_interval(current_estimate, shots_count, max_rounds, alpha_confidence_level):
    
    epsilon = sqrt(3 * log(2 * max_rounds / alpha_confidence_level) / shots_count)
    
    lower = max(0, current_estimate - epsilon)
    upper = min(1, current_estimate + epsilon)
    
    chernoff_confidence_interval = (lower, upper)

    # print(f'IQAE chernoff_confidence_interval: {chernoff_confidence_interval}')   
    
    return chernoff_confidence_interval


def clopper_pearson_confidence_interval(positive_counts, shots, alpha_confidence_level):
    
    lower, upper = 0, 1

    # if counts == 0, the beta quantile returns NaN
    
    if positive_counts != 0:
        
        lower = beta.ppf(q=alpha_confidence_level / 2,
                         a=positive_counts, 
                         b=shots - positive_counts + 1)

    # if counts == shots, the beta quantile returns NaN
    
    if positive_counts != shots:
        
        upper = beta.ppf(q=1 - alpha_confidence_level / 2,
                         a=positive_counts + 1,
                         b=shots - positive_counts)
        
    clopper_pearson_confidence_interval = (lower, upper)

    # print(f'IQAE clopper_pearson_confidence_interval: {clopper_pearson_confidence_interval}')  
    
    return clopper_pearson_confidence_interval


def find_next_power(power, is_upper_half_circle, theta_interval, min_power_increase_ratio):
    

    theta_lower, theta_upper = theta_interval
    
    old_scaling_factor = 4 * power + 2
    
    max_scaling_factor = int(1 / (2 * (theta_upper - theta_lower)))
    
    scaling_factor = max_scaling_factor - (max_scaling_factor - 2) % 4
    

    while scaling_factor >= min_power_increase_ratio * old_scaling_factor:
        
        theta_min = scaling_factor * theta_lower - int(scaling_factor * theta_lower)
        theta_max = scaling_factor * theta_upper - int(scaling_factor * theta_upper)
        
        if theta_min <= theta_max <= 0.5 and theta_min <= 0.5:
            
            is_upper_half_circle = True
            power = int((scaling_factor - 2) / 4)
            break

        elif theta_max >= 0.5 and theta_max >= theta_min >= 0.5:
            
            is_upper_half_circle = False
            power = int((scaling_factor - 2) / 4)
            break

        scaling_factor -= 4

    return power, is_upper_half_circle


def build_iqae_circuit(state_preparation,
                       grover_operator,
                       objective_qubits,
                       k=0,
                       measurement=False):
    
    all_qubits_count = max(state_preparation.num_qubits,
                           grover_operator.num_qubits)
                           
    measurement_bits_count = len(objective_qubits)
    
    iqae_register = QuantumRegister(all_qubits_count, 'iqae')
                     
    iqae_circuit = QuantumCircuit(iqae_register, name="IQAE")
    
    iqae_circuit.append(state_preparation, iqae_register)

    if k != 0:
        iqae_circuit.append(grover_operator.power(k), iqae_register)
        
    if measurement:
        
        measurement_register = ClassicalRegister(measurement_bits_count, 'measure')
        iqae_circuit.add_register(measurement_register)
        
        iqae_circuit.barrier()
        iqae_circuit.measure(iqae_register, measurement_register)

    return iqae_circuit
    

class BernoulliCircuit(QuantumCircuit):
    
    QUBIT_COUNT = 1

    def __init__(self, probability, circuit_type):
        
        super().__init__(1)
        
        self.name = 'Bernoulli ' + circuit_type.capitalize()
        
        self.theta_p = 2 * asin(probability ** 0.5)
        
        if circuit_type == 'diffuser':
            self.theta_p *= 2
            
        self.ry(self.theta_p, 0)
        

    def build_power_circuit(self, power):
        
        bernoulli_power_circuit = QuantumCircuit(1)
        bernoulli_power_circuit.name=f'{self.name} ** {power}'
        bernoulli_power_circuit.ry(power * self.theta_p, 0)
        
        return bernoulli_power_circuit
    
    power = build_power_circuit
        

### Main functions

def iqae(run_values, task_log):
    
    dummy_circuit = QuantumCircuit(1)
    dummy_circuit.measure_all()
    
    return dummy_circuit
    
    
def iqae_post_processing(run_data, task_log):
    
    """
    Create Iterative Quantum Amplitude Estimation (IQAE) circuit
    
    https://qiskit.org/documentation/finance/tutorials/00_amplitude_estimation.html
    
    """
    
    # Input data
    
    run_values = run_data.get('Run Values')
    
    input_bernoulli_probability = run_values.get('bernoulli_probability')
    input_epsilon = run_values.get('epsilon')
    input_alpha = run_values.get('alpha')
    confidence_interval_method = run_values.get('confidence_interval_method')
    
    bernoulli_probability = float(input_bernoulli_probability)
    epsilon = float(input_epsilon)
    alpha = float(input_alpha)

    min_power_increase_ratio = 2
    
    
    # Bernoulli Circuits
    
    bernoulli_operator = BernoulliCircuit(bernoulli_probability, circuit_type='operator')
    bernoulli_diffuser = BernoulliCircuit(bernoulli_probability, circuit_type='diffuser')
    
    controlled_bernoulli_diffuser = bernoulli_diffuser.control()
    controlled_bernoulli_diffuser.name = 'Controlled Bernoulli Diffuser'
    

    # Simulator
    
    shots = 1024
    backend = Aer.get_backend("aer_simulator")
    simulator = QuantumInstance(backend)

        
    # Initialization

    iteration_number = 0
    oracle_queries_count = 0

    powers = [0]
    multiplication_factors = []
    one_shots_counts = []
    
    theta_intervals = [[0, 1 / 4]]
    amplitude_intervals = [[0.0, 1.0]]

    good_qubits = [0]
    is_good_state = lambda x: all(bit == "1" for bit in x)

    is_upper_half_circle = True
    
    max_rounds = int(log(min_power_increase_ratio * pi / 8 / epsilon) / 
                     log(min_power_increase_ratio)) + 1
    
    theta_interval = theta_intervals[-1]
    theta_lower, theta_upper = theta_interval
    theta_difference = theta_upper - theta_lower
    
    
    # Iterations
    
    while theta_difference > epsilon / pi:
        
        iteration_number += 1
        
        last_power = powers[-1]

        power, is_upper_half_circle = find_next_power(
            last_power,
            is_upper_half_circle,
            theta_interval,
            min_power_increase_ratio=min_power_increase_ratio,
        )

        powers.append(power)
        
        multiplication_factor = (2 * power + 1) / (2 * last_power + 1)
        
        multiplication_factors.append(multiplication_factor)

        iqae_circuit = build_iqae_circuit(bernoulli_operator,
                                          bernoulli_diffuser,
                                          good_qubits,
                                          power,
                                          measurement=True)
                                     
        result = simulator.execute(iqae_circuit)
        
        counts = result.get_counts()

        one_counts = sum(state_counts for state, state_counts in counts.items()
                         if is_good_state(state))
                         
        counts_total = sum(counts.values())
        
        probability_of_measuring_one = one_counts / counts_total

        one_shots_counts.append(one_counts)

        oracle_queries_count += shots * power
        

        # Amplitude range

        if confidence_interval_method == "chernoff":
            
            amplitude_range = chernoff_confidence_interval(probability_of_measuring_one,
                                                           shots,
                                                           max_rounds,
                                                           alpha)
            
        elif confidence_interval_method == "clopper_pearson":
            
            alpha_confidence_level = alpha / max_rounds
            
            amplitude_range = clopper_pearson_confidence_interval(one_counts, 
                                                                  shots,
                                                                  alpha_confidence_level)
                                                                   
        amplitude_min, amplitude_max = amplitude_range
        

        # Theta
        
        if is_upper_half_circle:
            theta_min = acos(1 - 2 * amplitude_min) / 2 / pi
            theta_max = acos(1 - 2 * amplitude_max) / 2 / pi
        else:
            theta_min = 1 - acos(1 - 2 * amplitude_max) / 2 / pi
            theta_max = 1 - acos(1 - 2 * amplitude_min) / 2 / pi

        scaling = 4 * power + 2
        
        last_theta_interval = theta_intervals[-1]
        last_theta_lower, last_theta_upper = last_theta_interval
        
        theta_upper = (int(scaling * last_theta_upper) + theta_max) / scaling
        theta_lower = (int(scaling * last_theta_lower) + theta_min) / scaling

        theta_difference = theta_upper - theta_lower
        
        theta_interval = [theta_lower, theta_upper]
        
        theta_intervals.append(theta_interval)
        

        # Amplitude
        
        amplitude_upper = sin(2 * pi * theta_upper) ** 2
        amplitude_lower = sin(2 * pi * theta_lower) ** 2
        
        amplitude_interval = [amplitude_lower, amplitude_upper]
        amplitude_intervals.append(amplitude_interval)
        
        # task_log(f'IQAE iqae_circuit:\n{iqae_circuit}\n')


    # Estimate

    confidence_interval = amplitude_interval
    confidence_interval_lower, confidence_interval_upper = confidence_interval

    epsilon_estimated = (confidence_interval_upper - confidence_interval_lower) / 2
    
    amplitude_estimated = sum(confidence_interval) / 2

    
    # Logs
    
    task_log(f'\nIQAE Circuit:\n{iqae_circuit}\n')    
    
    task_log(f'IQAE Input data:\n')
    task_log(f'IQAE epsilon: {epsilon}')
    task_log(f'IQAE alpha: {alpha}')
    task_log(f'IQAE confidence_interval_method: {confidence_interval_method}')
    task_log(f'IQAE min_power_increase_ratio: {min_power_increase_ratio}\n')

    task_log(f'IQAE Processing:\n')    
    task_log(f'IQAE iteration_number: {iteration_number}')    
    task_log(f'IQAE oracle_queries_count: {oracle_queries_count}')
    task_log(f'IQAE one_shots_counts: {one_shots_counts}')
    task_log(f'IQAE multiplication_factors: {multiplication_factors}')
    task_log(f'IQAE powers: {powers}\n')

    task_log(f'IQAE theta_intervals: {theta_intervals}')    
    task_log(f'IQAE amplitude_intervals: {amplitude_intervals}')
    task_log(f'IQAE confidence_interval: {confidence_interval}\n')

    task_log(f'IQAE Results:\n') 
    task_log(f'IQAE epsilon_estimated: {epsilon_estimated}')
    task_log(f'IQAE amplitude_estimated: {amplitude_estimated}')
    
    
    return {'Estimated Amplitude': amplitude_estimated}
