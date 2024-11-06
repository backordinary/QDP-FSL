# https://github.com/smturro2/Portfolio-Picker/blob/554fcf55b11ee95eb8bde5d8c569e69dbb137d48/src/Find_Best_Strategy.py
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua import QuantumInstance
from qiskit import Aer
import pandas as pd
import numpy as np
from Portfolio_Manager import PortfolioManager
from Edited_Qiskit_portfolio import *
import re


def getBestStrategy(portfolioMatrix,risk,budget,penalty=0):
    # Set seed for reproducable results
    seed = 503678

    # Get the moments of the stocks
    df_Prices = pd.read_csv("DataFiles/StockPrices.csv")

    np_Prices = df_Prices.to_numpy()
    np_Returns = np_Prices[1:,:]/np_Prices[0:-1,:]-1
    if(np.isnan(np_Returns).any()):
        raise ValueError("Invalid Stock Data")

    mu = np.mean(np_Returns,axis=0)
    sigma = np.cov(np_Returns.T)

    # Set up the Quantum operator for a financial portfolio
    qubitOperator, offset = get_operator(mu, sigma, risk, portfolioMatrix, budget, penalty)

    # Set backend. We are considering pure states
    # so Aer's state vector simulator will work best
    backend = Aer.get_backend('statevector_simulator')

    # Set up optimization
    cobyla = COBYLA() # Constrained Optimization By Linear Approximation optimizer.
    cobyla.set_options(maxiter=250)
    qaoa = QAOA(qubitOperator, cobyla, 3) # Quantum Approximate Optimization Algorithm
    qaoa.random_seed = seed
    quantum_instance = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

    # Run optimization
    result = qaoa.run(quantum_instance)

    # Parse Results
    op_State = sample_most_likely(result.eigenstate)
    op_Cost = portfolio_value(op_State, mu, sigma, risk, portfolioMatrix, budget, penalty)
    op_Return = portfolio_expected_value(op_State, mu, portfolioMatrix)
    op_Risk = portfolio_variance(op_State, sigma, portfolioMatrix)

    return [op_State,op_Cost,op_Return,op_Risk]

def roundFloat(input1):
    return np.floor(input1*1000+.5)/1000

def buildStrategyHTML(optimization_results,portManager):
    df_ports = portManager.df_ports
    df_stocks = portManager.df_stocks
    strat = optimization_results[0]
    expReturn = roundFloat(optimization_results[2])
    risk = roundFloat(optimization_results[3])
    TORREPLACE = '<h2 id="StratResults">This strategy has a expected return of '+\
                  str(expReturn)+'% per day! and a predicted risk of '+str(risk)+'</h2>'
    TORREPLACE += '<h3 id="StratResults">Click on a portfolio to view its components.</h3>'

    TORREPLACE += '<table id="tableOfPorts">' \
                    '<tr id="headerRow">' \
                        '<th style="width: 100px;">Included</th>' \
                        '<th colspan="3">Portfolio Name</th>' \
                    '</tr>'
    for i in range(len(strat)):
        portName = df_ports['name'][i]
        className = re.sub(r"[ ']", '', portName)

        TORREPLACE += '<tr><td colspan="4"><table class="port" onclick="showDetails(\''+className+'\')">'
        # Row with checkmark and portfolio name
        TORREPLACE += '<tr>'
        if (abs(strat[i] - 1) < 10 ** -6):
            TORREPLACE += '<td style="border-right: .5px lightgray solid;text-align:center;width:75px;"><i style="color:#32a852;" class="fa fa-check" aria-hidden="true"></i></td>'
        else:
            TORREPLACE += '<td style="border-right: .5px lightgray solid;text-align: center;width:75px;"><i style="color:red;" class="fa fa-times" aria-hidden="true"></i></td>'

        TORREPLACE += '<td colspan="3" style="padding-left:25px;">'+portName+'</td>'
        TORREPLACE += '</tr>'

        # Collapsable rows of details
        TORREPLACE += '<tr id="headerRow" style="display:none;" class="' + className + '">'
        TORREPLACE += '<th>Ticker</th>'
        TORREPLACE += '<th>Name</th>'
        TORREPLACE += '<th>Sector</th>'
        TORREPLACE += '<th>Weight</th>'
        TORREPLACE += '</tr>'
        numTick = 0
        for tick in df_ports['list_of_stocks'][i]:
            index_list = df_stocks.index[df_stocks['Symbol'] == tick].tolist()
            if len(index_list) != 1:
                # todo : remove this
                raise ValueError("Invalid Stock in Portfolio")
            stock_index = index_list[0]

            TORREPLACE += '<tr class="'+className+'" style="display:none;">'
            TORREPLACE += '<td>'+str(df_stocks.loc[stock_index,"Symbol"])+'</td>'
            TORREPLACE += '<td>'+str(df_stocks.loc[stock_index,"Security"])+'</td>'
            TORREPLACE += '<td>'+str(df_stocks.loc[stock_index,"GICS Sector"])+'</td>'
            TORREPLACE += '<td>'+str(roundFloat(df_ports['weights'][i][numTick])*100)+'%</td>'
            TORREPLACE += '</tr>'
            numTick +=1

        TORREPLACE += '</table></td></tr>'
    TORREPLACE += '</table>'
    return TORREPLACE


# todo : get rid of this
if __name__ == "__main__":

    # Get portfolio matrix
    port_manager = PortfolioManager()
    portfolioMatrix = port_manager.getPortfolioMatrix()

    results = getBestStrategy(portfolioMatrix=portfolioMatrix,risk=.5,budget=1,penalty=2)
    print("Nice")
