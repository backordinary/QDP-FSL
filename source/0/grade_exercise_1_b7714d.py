# https://github.com/seyurlutchminarain/Quantum-Africa-Challenge/blob/85f5945906ddb83f77411e6f551929fa8edf2f3b/2_Intermediate/grade_exercise_1.py
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import IterativeAmplitudeEstimation
from qiskit_finance.circuit.library import LogNormalDistribution, EuropeanCallPricingObjective
from qiskit_finance.applications.estimation import EuropeanCallPricing

def grade_exercise_1(solutions):
    num_uncertainty_qubits = solutions[0]
    low = solutions[1]
    high = solutions[2]
    epsilon = solutions[3]
    alpha = solutions[4]
    shots = int(solutions[5])
    simulator = solutions[6]
    
    S = 50 
    strike_price = 55
    vol = 0.4     
    r = 0.05     
    T = 30 / 365  
    mu = ((r - 0.5 * vol**2) * T + np.log(S))
    sigma = vol * np.sqrt(T)
    mean = np.exp(mu + sigma**2/2)
    variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
    stddev = np.sqrt(variance)
    
    uncertainty_model = LogNormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high))
    
    european_call_pricing = EuropeanCallPricing(num_state_qubits=num_uncertainty_qubits,
                                            strike_price=strike_price,
                                            rescaling_factor=0.25,
                                            bounds=(low, high),
                                            uncertainty_model=uncertainty_model)
    
    qi = QuantumInstance(Aer.get_backend(simulator), shots=shots, seed_simulator=42, seed_transpiler=42)
    problem = european_call_pricing.to_estimation_problem()

    ae = IterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)
    result = ae.estimate(problem)
    
    x = uncertainty_model.values
    y = np.maximum(0, x - strike_price)
    exact_value = np.dot(uncertainty_model.probabilities, y)
    estimated_result = european_call_pricing.interpret(result)
    error = np.abs(exact_value-estimated_result)
    
    # Score result 
    
    if error <= 0.01:
        return print('Congrats! You achieved the best score of: ', 100)
    if error > 0.01 and error < 0.03:
        return print('Your score is: ', 50, '. Awesome work! But you can do better. Try fine tuning more parameters')
    elif error >= 0.03 and error < 0.04:
        return print('Your score is: ', 20, 'Good job, but try changing more parameters to get an estimation error less than 0.03!')
    else: 
        return print('Your score is: ', 0, ' try changing more parameters to get an estimation error less than 0.03!')
