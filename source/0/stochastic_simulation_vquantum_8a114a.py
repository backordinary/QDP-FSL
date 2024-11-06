# https://github.com/AlexanderKosmehl/qiskit-normal-distribution/blob/3e04b9d3295a629765812d8867c2df384b4c1108/stochastic_simulation_vQuantum.py
from qiskit import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.aqua.components.uncertainty_models import NormalDistribution
from qiskit.providers.aer import AerSimulator
sim = AerSimulator()
from collections import OrderedDict
from numpy.core.fromnumeric import std
from numpy.lib.function_base import average
import random
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Variable Setup
mu, sigma = 560, 200
num_qubits = 5
low = (mu - 3 * sigma)
high = (mu + 3 * sigma)

# Generate Circuit
q = QuantumRegister(num_qubits, 'q')
c = ClassicalRegister(num_qubits, 'c')
circuit = QuantumCircuit(q, c)

# Build Distribution Circuit
normal = NormalDistribution(num_target_qubits=num_qubits, mu=mu, sigma=sigma, low=low, high=high)
normal.build(circuit, q)
circuit.measure(q, c)

# Execute Simulation
job = sim.run(circuit)
counts = job.result().get_counts()

# Transform Results into correct Distribution
int_counts = {int(key, 2):value for (key, value) in counts.items()}
mapped_counts = {((high - low) / (2**num_qubits - 1) * number + low):occurences for number, occurences in int_counts.items()}
ordered_counts = OrderedDict(sorted(mapped_counts.items()))

# Transform Results into list to draw samples from
sample_list = []
for number, occurences in ordered_counts.items():
  for _ in range(occurences):
    sample_list.append(number)

# Check results
print(f"Average: {average(sample_list)}")
print(f"Std: {std(sample_list)}")

overCost = 1
underCost = 10
simulationStart = 500
simulationStop = 1100
iterationsPerSimulation = 100

# Helper Functions

# Generates a sample of any size from a normal distribution
# mu = mean, sigma = standard deviation
# sampleSize = numbers to be generated, default=1
def generateGaussianSample():
    return sample_list[random.randint(0, len(sample_list) - 1)]


# Evaluates the cost of the products based on the asymetrical cost function
# Too many products = 1€ per unnescessary product
# Too few products = 10€ per missed customer
def evaluatePrintedAmount(customer, amount):
    if customer >= amount:
        return (customer - amount) * underCost
    else:
        return (amount - customer) * overCost


# Simulates a given number of printed products any number of times and takes the average
# amount = amount of products to be printed
# times = simulation runs
def simulateAmount(amount, times=1):
    evaluations = [evaluatePrintedAmount(generateGaussianSample(), amount) for _ in range(0, times)]

    return average(evaluations).round().astype(int)


# Generates a list of all possible amounts of products
possibleAmounts = range(simulationStart, simulationStop)

# Simulates each of the given possible amounts
simulatedCosts = [simulateAmount(possibleAmount, iterationsPerSimulation) for possibleAmount in possibleAmounts]

# Performs a polynomial regression on the simulation results
polynomialModel = np.poly1d(np.polyfit(possibleAmounts, simulatedCosts, 3))
linespace = np.linspace(simulationStart, simulationStop - 1, max(simulatedCosts))

# Get coefficients from regression model
a, b, c, d = polynomialModel.coefficients
print(f"\nPolynom: f(x) = {a} x^3 + {b} x^2 + {c} x + {d}")

# Finds minimum of regression function
min = optimize.minimize(lambda x: a * x ** 3 + b * x ** 2 + c * x + d, x0=0)
minx = round(min.x[0], 2)
miny = round(min.fun, 2)
print(f"\nMinimum of {miny} at {minx} \n")

# Plots the results
plt.title("Simulation Results")
plt.xlabel("Printed Amount")
plt.ylabel("Expected Cost")
plt.scatter(possibleAmounts, simulatedCosts)
plt.plot(linespace, polynomialModel(linespace), color="r")
plt.annotate(f"Minimum of {miny} \nat {minx}",
             xy=(minx, miny),
             xytext=(0.6, 0.5),
             xycoords="data",
             textcoords="axes fraction",
             arrowprops=dict(facecolor="black", shrink=0.05),
             horizontalalignment="center")
plt.show()
