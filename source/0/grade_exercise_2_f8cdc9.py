# https://github.com/seyurlutchminarain/Quantum-Africa-Challenge/blob/85f5945906ddb83f77411e6f551929fa8edf2f3b/2_Intermediate/grade_exercise_2.py
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit.circuit.library import  LinearAmplitudeFunction

def grade_exercise_2(uncertainty_model, european_put_objective, ae):
    num_uncertainty_qubits = 4
    S = 200 # initial spot price
    vol = 0.3 # volatility of 40%
    r = 0.08 # annual interest rate of 4%
    T = 60 / 365 # 60 days to maturity
    strike_price = 230
    epsilon = 0.01
    alpha = 0.05
    shots = 100
    
    european_put = european_put_objective.compose(uncertainty_model, front=True)

    problem = EstimationProblem(state_preparation=european_put,
                            objective_qubits=[num_uncertainty_qubits],
                            post_processing=european_put_objective.post_processing)
    
    result = ae.estimate(problem)
    estimated_result = result.estimation_processed
    
    # true result:
    
    # set the approximation scaling for the payoff function
    rescaling_factor = 0.25

    # resulting parameters for log-normal distribution
    mu = ((r - 0.5 * vol**2) * T + np.log(S))
    sigma = vol * np.sqrt(T)
    mean = np.exp(mu + sigma**2/2)
    variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
    stddev = np.sqrt(variance)
    low  = np.maximum(0, mean - 2*stddev) 
    high = mean + 2*stddev
    breakpoints = [low, high]
    slopes = [-1, 0]
    offsets = [strike_price - low, 0]
    f_min = 0
    f_max = strike_price - low
    
    uncertainty_model = LogNormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high))
    
    european_put_objective = LinearAmplitudeFunction(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low, high),
    image=(f_min, f_max),
    rescaling_factor=rescaling_factor,
    breakpoints=breakpoints   
    )
    
    qi = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=100, seed_simulator=42, seed_transpiler=42)
    ae = IterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)
    
    european_put = european_put_objective.compose(uncertainty_model, front=True)

    problem = EstimationProblem(state_preparation=european_put,
                            objective_qubits=[num_uncertainty_qubits],
                            post_processing=european_put_objective.post_processing)
    
    result = ae.estimate(problem)
    correct_result = result.estimation_processed
    
    error = correct_result - estimated_result
    
    if error == 0.0:
        return 'Well done! You have successfully implemented a put option using a quantum algorithm! :)'
    else:
        return 'Unfortunately, you have not implemented the algorithm correctly. Please make sure all parameters are set to the correct specified values, the seeds for the simulator and transpiler are unchanged and try again.'
    
    
    
    
    
    