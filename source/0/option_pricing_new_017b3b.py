# https://github.com/SchmidtMoritz/QT2_Group_Challenge/blob/c5c1e1ffb0b33ff9eed363e4a4c55759ebddb302/option_pricing_new.py
#https://qiskit.org/documentation/tutorials/finance/index.html

import matplotlib.pyplot as plt
import numpy as np

from qiskit import Aer, QuantumCircuit

from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit.algorithms.amplitude_estimators import (IterativeAmplitudeEstimation,
                                                    FasterAmplitudeEstimation,
                                                    AmplitudeEstimation,
                                                    MaximumLikelihoodAmplitudeEstimation)
from qiskit_finance.applications import EuropeanCallPricing,EuropeanCallDelta

from qiskit import IBMQ



# number of qubits to represent the uncertainty
num_uncertainty_qubits = 2  # here we use m=n for simplicity

# parameters for considered random distribution
S = 2.0       # initial spot price
vol = 0.4     # volatility of 40%
r = 0.05      # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity

# resulting parameters for log-normal distribution
mu = ((r - 0.5 * vol**2) * T + np.log(S))
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2/2)
variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low  = np.maximum(0, mean - 3*stddev)
high = mean + 3*stddev

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
uncertainty_model = LogNormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high))

# plot probability distribution
x = uncertainty_model.values
y = uncertainty_model.probabilities
plt.bar(x, y, width=0.2)
plt.xticks(x, size=15, rotation=90)
plt.yticks(size=15)
plt.grid()
plt.xlabel('Spot Price at Maturity $S_T$ (\$)', size=15)
plt.ylabel('Probability ($\%$)', size=15)
plt.show()

# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 1.896

# set the approximation scaling for the payoff function
c_approx = 0.25

# setup piecewise linear objective fcuntion
breakpoints = [low, strike_price]
slopes = [0, 1]
offsets = [0, 0]
f_min = 0
f_max = high - strike_price

# plot exact payoff function (evaluated on the grid of the uncertainty model)
x = uncertainty_model.values
y = np.maximum(0, x - strike_price)
plt.plot(x, y, 'ro-')
plt.grid()
plt.title('Payoff Function', size=15)
plt.xlabel('Spot Price', size=15)
plt.ylabel('Payoff', size=15)
plt.xticks(x, size=15, rotation=90)
plt.yticks(size=15)
plt.show()

# evaluate exact expected value (normalized to the [0, 1] interval)
exact_value = np.dot(uncertainty_model.probabilities, y)
exact_delta = sum(uncertainty_model.probabilities[x >= strike_price])



# generate european call pricing problem and transform it to an amplitude estimation problem
ecp = EuropeanCallPricing(num_uncertainty_qubits,strike_price,c_approx,(low,high),uncertainty_model)
ecp_est_problem = ecp.to_estimation_problem()

# call delta is the probability that the payoff is >0
# generate european call delta problem and transform it to an amplitude estimation problem
ecd = EuropeanCallDelta(num_uncertainty_qubits,strike_price,(low,high),uncertainty_model)
ecd_est_problem = ecd.to_estimation_problem()

IBMQ.load_account()
provider=IBMQ.get_provider('ibm-q')
#print(provider.backends())
#backend = provider.get_backend('ibmq_lima')
backend = Aer.get_backend('qasm_simulator')

# set target precision and confidence level
epsilon = 0.01
alpha = 0.05

iae = IterativeAmplitudeEstimation(epsilon,alpha,quantum_instance=backend)

res_ecp = iae.estimate(ecp_est_problem)
res_ecd = iae.estimate(ecd_est_problem)

#uncomment any of the following blocks to use a different amplitude estimation algorithm
'''
ae = AmplitudeEstimation(num_uncertainty_qubits,quantum_instance=backend)

res_ecp = ae.estimate(ecp_est_problem)
res_ecd = ae.estimate(ecd_est_problem)
'''

'''
fae = FasterAmplitudeEstimation(0.01,8,quantum_instance=backend)

res_ecp = fae.estimate(ecp_est_problem)
res_ecd = fae.estimate(ecd_est_problem)
'''

'''
mlae = MaximumLikelihoodAmplitudeEstimation(4,quantum_instance=backend)

res_ecp = mlae.estimate(ecp_est_problem)
res_ecd = mlae.estimate(ecd_est_problem)
'''

pricing_estimate = ecp.interpret(res_ecp)
delta_estimate = ecd.interpret(res_ecd)

print("Price:")
print(f'Exact value:        \t{exact_value}')
print(f'Estimated value:    \t{pricing_estimate}')
print("Delta:")
print(f'Exact value:        \t{exact_delta}')
print(f'Estimated value:    \t{delta_estimate}')
