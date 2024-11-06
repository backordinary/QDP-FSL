# https://github.com/huynh-alex/quantum-stocks/blob/5f2bcda5fb2e439ac2fff077e4e600a28bb50ac9/server/models/analyze.py
"""
This code is a part of Qiskit
© Copyright IBM 2017, 2021.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
"""

import sqlite3
import requests
from qiskit import Aer
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
from typing import cast
import time

class Analyze:

    def __init__(self):
        self.api_key = 'D4O43NY0RLJLTQ5B'
        self.db = sqlite3.connect('server/databases/stocks.db', check_same_thread=False)
        self.daily_closings = []
        self.daily_closings_obtained = 0
        self.progress = ''
        self.qp = None

    def divide(self, val_1, val_2):
        if val_2 == 0 and val_1 == 0:
            return 1
        if val_2 == 0:
            return np.nan
        return val_1 / val_2

    def get_period_return_mean_vector(self, daily_prices):
        div_func = np.vectorize(self.divide)
        period_returns = div_func(np.array(daily_prices)[:, 1:], np.array(daily_prices)[:, :-1]) - 1
        return cast(np.ndarray, np.mean(period_returns, axis=1))

    def get_period_return_covariance_matrix(self, daily_prices):
        div_func = np.vectorize(self.divide)
        period_returns = div_func(np.array(daily_prices)[:, 1:], np.array(daily_prices)[:, :-1]) - 1
        return np.cov(period_returns)

    def get_daily_prices(self, selectedStocks):
        self.daily_closings = []
        cur = self.db.cursor()
        self.daily_closings_obtained = 0
        daily_closings = []
        for company in selectedStocks:
            ticker = cur.execute('SELECT ticker FROM stocks WHERE company = (?)', (company,)).fetchone()[0]
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=' + ticker + '&apikey=' + self.api_key
            r = requests.get(url)
            data = r.json()
            
            if 'Note' in data:
                print('Waiting for about 1 minute.')
                self.progress = 'Fetching daily time series. ' + str(self.daily_closings_obtained) + '/' + str(len(selectedStocks)) + ' obtained.'
                time.sleep(61)
                r = requests.get(url)
                data = r.json()                
            time_series_daily_dict = data['Time Series (Daily)']
            self.progress = 'Fetching daily time series. ' + str(self.daily_closings_obtained) + '/' + str(len(selectedStocks)) + ' obtained.'
            time_series_daily_list = []
            for date in time_series_daily_dict:
                time_series_daily_list.append(time_series_daily_dict[date]['5. adjusted close'])
            daily_closings.append(time_series_daily_list)    
            self.daily_closings_obtained += 1

        for i in range(len(daily_closings)): #convert all strings to floats
            daily_closings[i] = list(map(float, daily_closings[i]))    
        
        return daily_closings

    def classical_optimization(self, selectedStocks, budget):
        self.daily_closings = self.get_daily_prices(selectedStocks)
        mu = self.get_period_return_mean_vector(self.daily_closings)
        sigma = self.get_period_return_covariance_matrix(self.daily_closings)

        q = 1              # risk factor
        # num_assets = len(selectedStocks)
        # penalty = num_assets      # parameter to scale the budget penalty term

        portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget)
        qp = portfolio.to_quadratic_program()
        self.qp = qp
        exact_mes = NumPyMinimumEigensolver()
        exact_eigensolver = MinimumEigenOptimizer(exact_mes)
        self.progress = 'Ongoing classical optimization.'
        result = exact_eigensolver.solve(qp)
        stocks_to_buy = result.x
        stocks_to_buy_list = []
        print(stocks_to_buy)
        for i in range(len(stocks_to_buy)):
            if stocks_to_buy[i] == 1:
                stocks_to_buy_list.append(selectedStocks[i])
        print(stocks_to_buy_list)
        return stocks_to_buy_list

    def quantum_optimization(self, selectedStocks):
        backend = Aer.get_backend('statevector_simulator')
        cobyla = COBYLA()
        cobyla.set_options(maxiter=500)
        ry = TwoLocal(len(self.daily_closings), 'ry', 'cz', reps=3, entanglement='full')
        quantum_instance = QuantumInstance(backend=backend, seed_simulator=1234, seed_transpiler=1234)
        vqe_mes = VQE(ry, optimizer=cobyla, quantum_instance=quantum_instance)
        vqe = MinimumEigenOptimizer(vqe_mes)
        self.progress = 'Ongoing quantum optimization.'
        result = vqe.solve(self.qp)
        stocks_to_buy = result.x
        stocks_to_buy_list = []
        for i in range(len(stocks_to_buy)):
            if stocks_to_buy[i] == 1:
                stocks_to_buy_list.append(selectedStocks[i])
        return stocks_to_buy_list

    def get_progress(self):
        return self.progress