# https://github.com/keamanansiber/qiskit/blob/6f1459aa130100251201ea8f0a7d78f97134919e/3QuantumMachineLearning/Tutorial/QGAN%202%20-%20Option%20Pricing%20with%20qGANs.py
import matplotlib.pyplot as plt
import numpy as np

from qiskit.circuit import ParameterVector
from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
from qiskit.finance.components.uncertainty_problems import EuropeanCallExpectedValue
from qiskit.aqua.components.uncertainty_models import UnivariateVariationalDistribution, NormalDistribution
from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.initial_states import Custom

from qiskit.aqua import aqua_globals, QuantumInstance

from qiskit import Aer


# Set upper and lower data values
bounds = np.array([0.,7.])
# Set number of qubits used in the uncertainty model
num_qubits = [3]

# Set entangler map
entangler_map = []
for i in range(sum(num_qubits)):
    entangler_map.append([i, int(np.mod(i+1, sum(num_qubits)))])

# Load the trained circuit parameters
g_params = [0.29399714, 0.38853322, 0.9557694, 0.07245791, 6.02626428, 0.13537225]

# Set an initial state for the generator circuit
init_dist = NormalDistribution(sum(num_qubits), mu=1., sigma=1., low=bounds[0], high=bounds[1])
init_distribution = np.sqrt(init_dist.probabilities)
init_distribution = Custom(num_qubits=sum(num_qubits), state_vector=init_distribution)

# construct the variational form
var_form = RealAmplitudes(sum(num_qubits), entanglement=entangler_map, reps=1, initial_state=init_distribution)
var_form.entanglement_blocks = 'cz'
theta = ParameterVector('θ', var_form.num_parameters)
var_form = var_form.assign_parameters(theta)

# Set generator circuit
g_circuit = UnivariateVariationalDistribution(sum(num_qubits), var_form, g_params,
                                              low=bounds[0], high=bounds[1])
g_circuit._var_form_params = theta

# construct circuit factory for uncertainty model
uncertainty_model = g_circuit


# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 2

# set the approximation scaling for the payoff function
c_approx = 0.25

# construct circuit factory for payoff function
european_call = EuropeanCallExpectedValue(
    uncertainty_model,
    strike_price=strike_price,
    c_approx=c_approx
)


# Evaluate trained probability distribution
uncertainty_model.set_probabilities(QuantumInstance(Aer.get_backend('statevector_simulator')))
x = uncertainty_model.values
y = uncertainty_model.probabilities

# Sample from target probability distribution
N = 100000
log_normal = np.random.lognormal(mean=1, sigma=1, size=N)
log_normal = np.round(log_normal)
log_normal = log_normal[log_normal <= 7]
log_normal_samples = []
log_normal_samples += [np.sum(log_normal==0)]
log_normal_samples += [np.sum(log_normal==1)]
log_normal_samples += [np.sum(log_normal==2)]
log_normal_samples += [np.sum(log_normal==3)]
log_normal_samples += [np.sum(log_normal==4)]
log_normal_samples += [np.sum(log_normal==5)]
log_normal_samples += [np.sum(log_normal==6)]
log_normal_samples += [np.sum(log_normal==7)]
log_normal_samples = np.array(log_normal_samples / sum(log_normal_samples))

# Plot distributions
plt.bar(x, y, width=0.2, label='trained distribution', color='royalblue')
plt.xticks(x, size=15, rotation=90)
plt.yticks(size=15)
plt.grid()
plt.xlabel('Spot Price at Maturity $S_T$ (\$)', size=15)
plt.ylabel('Probability ($\%$)', size=15)
plt.plot(log_normal_samples,'-o', color ='deepskyblue', label='target distribution', linewidth=4, markersize=12)
plt.legend(loc='best')
plt.show()


# Evaluate payoff for different distributions
payoff = np.array([0,0,0,1,2,3,4,5])
ep = np.dot(log_normal_samples, payoff)
print("Analytically calculated expected payoff w.r.t. the target distribution:  %.4f" % ep)
ep_trained = np.dot(y, payoff)
print("Analytically calculated expected payoff w.r.t. the trained distribution: %.4f" % ep_trained)

# Plot exact payoff function (evaluated on the grid of the trained uncertainty model)
x = uncertainty_model.values
y_strike = np.maximum(0, x - strike_price)
plt.plot(x, y_strike, 'ro-')
plt.grid()
plt.title('Payoff Function', size=15)
plt.xlabel('Spot Price', size=15)
plt.ylabel('Payoff', size=15)
plt.xticks(x, size=15, rotation=90)
plt.yticks(size=15)
plt.show()


# set target precision and confidence level
epsilon = 0.01
alpha = 0.05

# construct amplitude estimation
ae = IterativeAmplitudeEstimation(epsilon=epsilon, alpha=alpha, a_factory=european_call)


result = ae.run(quantum_instance=Aer.get_backend('qasm_simulator'), shots=100)


conf_int = np.array(result['confidence_interval'])
print('Exact value:        \t%.4f' % ep_trained)
print('Estimated value:    \t%.4f' % (result['estimation']))
print('Confidence interval:\t[%.4f, %.4f]' % tuple(conf_int))


import qiskit.tools.jupyter
#%qiskit_version_table
#%qiskit_copyright


# Confidence interval:	[0.9822, 1.0430]

# sumber = https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/finance/10_qgan_option_pricing.ipynb
