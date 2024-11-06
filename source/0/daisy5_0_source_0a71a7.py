# https://github.com/CalvinGreen94/lafrancapi/blob/9af90a5b98d1fa770dcdae8631acdb9659c9012e/LaFrancChainAPI/dAIsy/dAIsy5.0_source.py
from agent2.agent import Agent
from functions import *
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from tronpy import Tron
from tronpy.keys import PrivateKey
from tronpy import Tron, Contract
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
# from sklearn import 
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,TimeSeriesSplit
import time
from sklearn.preprocessing import MinMaxScaler
import datetime
import numpy as np
import matplotlib.dates as mdates
# %matplotlib inline
import ssl
import json
import ast
import os
import yfinance as yf
# client = Tron()  # The default provicder, mainnet
client = Tron()
#client = Tron(network="nile")  # The nile Testnet is preset
# or "nile", "tronex"
from qiskit import IBMQ
IBMQ.save_account('')
# IBMQ.disable_account()
IBMQ.disable_account()
IBMQ.enable_account('')
# IBMQ.backends()
providers = IBMQ.providers()
providers
provider = IBMQ.get_provider(hub='ibm-q') 
provider
provider.backends()
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
psi = random_statevector(2)
init_gate = Initialize(psi)
init_gate.label = "init"
backend = provider.get_backend('ibmq_manila')
def create_bell_pair(qc, a, b):
    """Creates a bell pair in qc using qubits a & b"""
    qc.h(a) # Put qubit a into state |+>
    qc.cx(a,b) # CNOT with a as control and b as target

qr = QuantumRegister(3, name="q")
crz, crx = ClassicalRegister(1, name="crz"), ClassicalRegister(1, name="crx")
teleportation_circuit = QuantumCircuit(qr, crz, crx)
create_bell_pair(teleportation_circuit, 1, 2)
inverse_init_gate = init_gate.gates_to_uncompute()
def alice_gates(qc, psi, a):
    qc.cx(psi, a)
    qc.h(psi)    
def new_bob_gates(qc, a, b, c):
    qc.cx(b, c)
    qc.cz(a, c)
qc = QuantumCircuit(3,1)

# First, let's initialize Alice's q0
qc.append(init_gate, [0])
qc.barrier()

# Now begins the teleportation protocol
create_bell_pair(qc, 1, 2)
qc.barrier()
# Send q1 to Alice and q2 to Bob
alice_gates(qc, 0, 1)
qc.barrier()
# Alice sends classical bits to Bob
new_bob_gates(qc, 0, 1, 2)

# We undo the initialization process
qc.append(inverse_init_gate, [2])
# Alice sends classical bits to Bob
new_bob_gates(qc, 0, 1, 2)

# We undo the initialization process
qc.append(inverse_init_gate, [2])
t_qc = transpile(qc, backend, optimization_level=2)
from tronpy.providers import HTTPProvider

# client = Tron(HTTPProvider("http://127.0.0.1:8678"))  # Use private network as HTTP API endpoint
# print(client.get_block())
# print(client.get_latest_block_number())
# print(client.get_latest_block_id())
data1=pd.read_csv('data/TRX-USD.csv') 

# data1 =data1[:data1.shape[0]] 
print(data1.shape[0])
bid_values = np.random.uniform(0.056,0.10,[data1.shape[0],1])
bid_df = pd.DataFrame(bid_values,columns=['Bid'])
bid =bid_df[-1:]
stake_values = np.random.uniform(0.025,0.08,[data1.shape[0],1])
stake_df = pd.DataFrame(stake_values,columns=['Stake'])
stake = stake_df[-1:]
clean_values = np.random.uniform(0.015,0.025,[data1.shape[0],1])
clean_df = pd.DataFrame(clean_values,columns=['Deck1'])
nsfw_values = np.random.uniform(0.026,0.04,[data1.shape[0],1])  
nsfw_df = pd.DataFrame(nsfw_values,columns=['Deck2']) 
damage1_values = np.random.randint(0,8,[data1.shape[0],1]) 
damage_df = pd.DataFrame(damage1_values,columns=['Deck1_damage'])
player_payout =  np.random.uniform(0.00001278,0.00003195,[data1.shape[0],1]) 
player_df = pd.DataFrame(player_payout,columns=['player_payout']) 
tx_fee = np.random.uniform(0.000000,0.00000,[data1.shape[0],1]) 
tx_df = pd.DataFrame(tx_fee,columns=['tx_fee']) 
damage2_values = np.random.randint(1,9,[data1.shape[0],1]) 
damage2_df = pd.DataFrame(damage2_values,columns=['Deck2_damage']) 

data = clean_df.join(nsfw_df)
data = data.join(damage_df) 
data = data.join(damage2_df) 
clean_image_no = np.random.randint(0,94,[data1.shape[0],1]) 
clean_image_df = pd.DataFrame(clean_image_no,columns=['clean_image_no'])
nsfw_image_no = np.random.randint(0,24,[data1.shape[0],1]) 
nsfw_image_df = pd.DataFrame(nsfw_image_no,columns=['nsfw_image_no']) 
data = data.join(clean_image_df) 
data = data.join(nsfw_image_df)
data=data.join(tx_df) 
data = data.join(player_df) 
data = data.join(bid_df) 
data = data.join(stake_df)
# data = data.to_csv('c_force_data.csv',index=False)
import os 
# PATH FOR PRICE-DECK-TRADE
# clean_path = os.listdir("images/media/anihotime/clean/")
data1 = data1.join(data) 
data1 = data1.drop(['Date'],axis=1) 
data1.to_csv('data/c_force_trx.csv')
# clean_path = np.array(clean_path)
# filename = clean_path[-1:]
data2=pd.read_csv('data/c_force_trx.csv')  
data2 = data2.drop(['Unnamed: 0'], axis=1)
print(data2) 
data2=data2.fillna(0.063)
print(client.get_account_balance(''))

priv_key = PrivateKey(bytes.fromhex(""))
# client.generate_address()

from tronpy import Tron
from tronpy.providers import HTTPProvider

# client = Tron(network='nile')
print(client.get_node_info())

for f in contract.functions:
    print(f)  # prints function signature(i.e. type info)

# # precision = cntr.functions.decimals()
# print('Balance in LaFrancCoins:{} LFR'.format(cntr.functions.equity_in_lafranccoins('TBeDPd2zP3piD8zBXfRfXFwhBAHYcUVPgy')))
# print('Balance in USD: ${}'.format(cntr.functions.equity_in_USD('TBeDPd2zP3piD8zBXfRfXFwhBAHYcUVPgy')))

import os
import csv
import datetime
import tensorflow as tf 
os.environ['KERAS_BACKEND' ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
daisy = 'dAIsy'
from matplotlib import style
import pandas as pd
if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	exit()
stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 710

import json

mint_acct = ''
print(''.format(client.get_account_balance('')))

priv_key = PrivateKey(bytes.fromhex(""))
for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)
	total_profit = 0
	agent.inventory = []
	starting_balance = client.get_account_balance(mint_acct)
	for t in range(l):
		action = agent.act(state)
		# hold
		next_state = getState(data, t + 1, window_size + 1)
		reward = 1.
		#buy
		if action == 1 :
			X=data2.drop(['Deck2_damage'],axis=1)
			y=data2['Deck2_damage']
			mini = MinMaxScaler()
		# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.13)
			from sklearn.linear_model import LinearRegression,LogisticRegression
			regressor = LinearRegression()

		#Fitting model with trainig data
			regressor.fit(X_train, y_train)

			y = regressor.predict(X_test[-1:]) 
			y = pd.DataFrame(y)
			print('PREDICTED Deck2 damage ',y)		
			print('teleporting from bob to alice')

			job = backend.run(t_qc)
			job_monitor(job)  # displays job status under cell	
			exp_result = job.result()
			exp_counts = exp_result.get_counts(qc)
			print(exp_counts)
			print('Sending TRX to LFR')
			print(' {}'.format(client.get_account_balance('')))

			txn = (client.trx.transfer(mint_acct,'', 1_000)
				.memo("")
				.build()
				.sign(priv_key)
				)
			txn.broadcast()

		elif action == 2 and len(agent.inventory) > 0:
			X=data2.drop(['Deck2_damage'],axis=1)
			y=data2['Deck2_damage']
			mini = MinMaxScaler()
			from sklearn.linear_model import LinearRegression,LogisticRegression
			regressor = LinearRegression()

		#Fitting model with trainig data
			regressor.fit(X_train, y_train)

			y = regressor.predict(X_test[-1:]) 
			y = pd.DataFrame(y)
			print('PREDICTED Deck2 damage ',y)					
			print('Sending TRX to LFRD')
			mint_acct = ''
			priv_key = PrivateKey(bytes.fromhex(""))
			print('PREDICTED Deck2 damage ',y)		
			print('teleporting from bob to alice')
			job = backend.run(t_qc)
			job_monitor(job)  # displays job status under cell	
			exp_result = job.result()
			exp_counts = exp_result.get_counts(qc)
			print(exp_counts)
			print(' {}'.format(client.get_account_balance('')))

			txn = (client.trx.transfer(mint_acct,'', 1_000)
				.memo("")
				.build()
				.sign(priv_key)
				)
			txn.broadcast()
		# a2 = pd.to_csv('SellPrice.csv')
		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))

		state = next_state
		s = pd.DataFrame(state)
		s = s.to_csv('state.csv')
		if done:
			print("--------------------------------")
			# print("Current info: ")
			print("--------------------------------")
			# print('Current Balance {}'.format(dai.functions.balanceOf(mint_acct).call()))

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		log_dir = "models/agent3/logs/" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
		agent.model.save("models/agent3/model_ep-" + str(e))
app.run(host = '127.0.0.1', debug=True,port=8678)
