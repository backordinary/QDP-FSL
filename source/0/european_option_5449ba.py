# https://github.com/VRSupriya/Quantum-Computing-Option-Pricing/blob/a4c4b51a049b5887e3067b444e248a831f69fa22/code/european_option.py
from random import seed
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.algorithms import AmplitudeEstimation
# from qiskit_finance.circuit.library import EuropeanCallPricingObjective #F
from qiskit_finance.circuit.library import LogNormalDistribution
# from qiskit.circuit.library import LogNormalDistribution #P_X
# from qiskit_finance.applications import EuropeanCallPricing
from qiskit_finance.applications.estimation import EuropeanCallPricing
from qiskit.algorithms import IterativeAmplitudeEstimation
from qiskit_finance.applications.estimation import EuropeanCallDelta

import pandas as pd
import datetime as dt
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import time
import pandas_datareader as web

# np.ranom.seed(0)
data= web.DataReader('AAPL',data_source='yahoo',start='1-1-2020',end='1-1-2021')['Adj Close']
# data=pd.DataFrame(data)
log_return= np.log(1+data.pct_change())
voltality=round(log_return.std(),2)
np.random.seed(10)

def get_option_motecarlo_vectorization(startdate_index, number_of_days, strike_price):
    sigma=voltality
    rf=0.05
    iterations=1000000
    T= number_of_days/252
    S0=data[startdate_index]
    #time
    start=time.time()
    option_data=np.zeros([iterations,2])
    #one array containg no_iterations observations with mean 0 and variance 1 within the range [0,1]
    rand=np.random.normal(0,1,[1,iterations])
    stock_price= S0*np.exp(T*(rf-0.5*sigma**2)+(sigma*np.sqrt(T)*rand))
    # stock_price = np.array(stock_price)
    option_data[:,1]=stock_price-strike_price
    average=np.sum(np.amax(option_data,axis=1)/float(iterations))
    presentValue_option= np.exp(-1.0*rf*T)*average  #present value of the option data by considerig time value of money
    actual_stock_price=data[startdate_index+number_of_days]
    time_taken=time.time()-start
    # 'option_motecarlo':average,
    return {'Execution_time_montecarlo':time_taken}

def get_option_motecarlo_forloop(startdate_index, number_of_days, strike_price):
    start=time.time()
    sigma=voltality
    rf=0.05
    iterations=1000000
    T= number_of_days/252
    S0=data[startdate_index]
    #time
    
    # option_data=np.zeros([iterations,2])
    #one array containg no_iterations observations with mean 0 and variance 1 within the range [0,1]
    rand=np.random.normal(0,1,[iterations])
    print(rand)
    # s(t)=s(o)e^[(r-1/2*sigma square)t+sigma*sqrt(t)*random values]
    option_data = []
    for i in range(iterations):
        option_data.append((S0*np.exp(T*(rf-0.5*sigma**2)+(sigma*np.sqrt(T)*rand[i])))-strike_price)
    
    # stock_price= S0*np.exp(T*(rf-0.5*sigma**2)+(sigma*np.sqrt(T)*rand))
    # stock_price = np.array(stock_price)
    # option_data[:,1]=stock_price-strike_price
    # average=np.sum(np.amax(option_data,axis=1)/float(iterations))
    average = np.sum(option_data)/float(iterations)
    presentValue_option= np.exp(-1.0*rf*T)*average  #present value of the option data by considerig time value of money
    actual_stock_price=data[startdate_index+number_of_days]
    time_taken=time.time()-start
    # 
    return {'option_motecarlo':average,'Execution_time_montecarlo':time_taken}


def get_option_amplitudeestimation_delta(start_date_index, number_of_days, strike_price):
    to = time.time()
    num_uncertainty_qubits = 3

    T = number_of_days / 252 

    S = data[start_date_index]
    r = 0.05   

    mu = ((r - 0.5 * voltality**2) * T + np.log(S))
    sigma = voltality * np.sqrt(T)
    mean = np.exp(mu + sigma**2/2)
    variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
    stddev = np.sqrt(variance)

    # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
    low  = np.maximum(0, mean - 3*stddev)
    high = mean + 3*stddev

    # construct A operator for QAE for the payoff function by
    # composing the uncertainty model and the objective
    
    uncertainty_model = LogNormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high))

    

    european_call_delta = EuropeanCallDelta(
        num_state_qubits=num_uncertainty_qubits,
        strike_price=strike_price,
        bounds=(low, high),
        uncertainty_model=uncertainty_model,
    )
    european_call_delta_circ = QuantumCircuit(european_call_delta._objective.num_qubits)
    european_call_delta_circ.append(uncertainty_model, range(num_uncertainty_qubits))
    european_call_delta_circ.append(
        european_call_delta._objective, range(european_call_delta._objective.num_qubits)
    )
    
    epsilon = 0.01
    alpha = 0.05

    qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=100)
    problem = european_call_delta.to_estimation_problem()
    ae_delta = IterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)
    result_delta = ae_delta.estimate(problem)

    
    return { "Option_qiskit": european_call_delta.interpret(result_delta),"Execution_time":time.time() - to}

def get_option_amplitudeestimation(start_date_index, number_of_days, strike_price):

    start = time.time()
    num_uncertainty_qubits = 3
    T = number_of_days / 252
    S = data[start_date_index]
    r = 0.05            # annual interest rate of 4%



    # resulting parameters for log-normal distribution
    mu = ((r - 0.5 * voltality**2) * T + np.log(S))
    sigma = voltality * np.sqrt(T)
    mean = np.exp(mu + sigma**2/2)
    variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
    stddev = np.sqrt(variance)

    # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
    low  = np.maximum(0, mean - 3*stddev)
    high = mean + 3*stddev
    print(strike_price,low,high)

    # construct A operator for QAE for the payoff function by
    # composing the uncertainty model and the objective
    
    uncertainty_model = LogNormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high))

    europian_call_pricing = EuropeanCallPricing(num_state_qubits=num_uncertainty_qubits, 
                                            strike_price=strike_price, 
                                            rescaling_factor=0.25, #approximation constant for 
                                            bounds=(low, high),
                                            uncertainty_model=uncertainty_model)
    problem = europian_call_pricing.to_estimation_problem()

    # set target precision and confidence level
    epsilon = 0.01 #determines final accuracy
    alpha = 0.05 #determines how certain we are of that result
    qi = QuantumInstance(Aer.get_backend('aer_simulator'), shots=100)
    
    ae = IterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)
    
    result = ae.estimate(problem)
    
   
    return {"Option_qiskit": result.estimation_processed, "Execution_time":time.time() - start}

# def get_option(start_date_index, number_of_days, strike_price):
#     print(start_date_index, number_of_days, strike_price)
#     result_montecarlo = get_option_motecarlo1(start_date_index, number_of_days, strike_price)
#     result_amplitudeestimation = get_option_amplitudeestimation1(start_date_index, number_of_days, strike_price)

# return {**result_montecarlo, **result_amplitudeestimation}
# result_monte_for = get_option_motecarlo_forloop(0,40,73)
# print(f"result of monte carlo using for loop:\t{result_monte_for}")

if __name__=="__main__":
     
    result_monte_vect = get_option_motecarlo_vectorization(0, 40, 73)
    print(f"result of monte-carlo using vectorization:\t{result_monte_vect}")

    result_amplitudeestimation = get_option_amplitudeestimation(0, 40, 73)
    print(f"result of QAE using phase estimation:\t{result_amplitudeestimation}")

    result_qae_delta= get_option_amplitudeestimation_delta(0,40,73)
    print(f"result of QAE using delta:\t{result_qae_delta}")

    

    