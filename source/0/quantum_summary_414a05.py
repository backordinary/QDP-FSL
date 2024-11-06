# https://github.com/betisb/BigDataAnalytics/blob/2939d2088c4648d2a35fdb76dba162582643da6b/src/Quantum_summary.py
from qiskit import *
import pandas as pd
import numpy as np
from qiskit import BasicAer, Aer, execute
from qiskit.visualization import plot_histogram


# this part(uf) will mark the frequency negative so that the diffuer V can keep amplifying the frequency to single a corresponding state


# n = 5
def phase_oracle000(n, name='Uf001'):
    qc = QuantumCircuit(n)
    qc.x(0)
    qc.x(1)
    qc.x(2)
    qc.ccx(0, 1, 3)
    qc.ccx(2, 3, 4)
    qc.ccx(0, 1, 3)
    qc.x(0)
    qc.x(1)
    qc.x(2)
    # display(qc.draw('mpl'))
    return qc


def phase_oracle001(n, name='Uf001'):
    qc = QuantumCircuit(n, name=name)
    qc.x(1)
    qc.x(2)
    qc.ccx(0, 1, 3)
    qc.ccx(2, 3, 4)
    qc.ccx(0, 1, 3)
    qc.x(1)
    qc.x(2)
    # qc.ccx(0,1,3)
    # qc.ccx(2,3,4)
    # qc.ccx(0,1,3)
    # display(qc.draw('mpl'))
    return qc


def phase_oracle010(n, name='Uf010'):
    qc = QuantumCircuit(n, name=name)
    qc.x(0)
    qc.x(2)
    qc.ccx(0, 1, 3)
    qc.ccx(2, 3, 4)
    qc.ccx(0, 1, 3)
    qc.x(0)
    qc.x(2)
    # display(qc.draw('mpl'))
    return qc


def phase_oracle011(n, name='Uf000'):
    qc = QuantumCircuit(n, name=name)
    qc.x(2)
    qc.ccx(0, 1, 3)
    qc.ccx(2, 3, 4)
    qc.ccx(0, 1, 3)
    qc.x(2)
    # qc.ccx(0,1,3)
    # qc.ccx(2,3,4)
    # qc.ccx(0,1,3)
    # display(qc.draw('mpl'))
    return qc


def phase_oracle100(n, name='Uf000'):
    qc = QuantumCircuit(n, name=name)
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 3)
    qc.ccx(2, 3, 4)
    qc.ccx(0, 1, 3)
    qc.x(0)
    qc.x(1)
    # display(qc.draw('mpl'))
    return qc


def phase_oracle101(n, name='Uf000'):
    qc = QuantumCircuit(n, name=name)
    qc.x(1)
    qc.ccx(0, 1, 3)
    qc.ccx(2, 3, 4)
    qc.ccx(0, 1, 3)
    qc.x(1)
    # qc.ccx(0,1,3)
    # qc.ccx(2,3,4)
    # qc.ccx(0,1,3)
    # display(qc.draw('mpl'))
    return qc


def phase_oracle110(n, name='Uf000'):
    qc = QuantumCircuit(n, name=name)
    qc.x(0)
    qc.ccx(0, 1, 3)
    qc.ccx(2, 3, 4)
    qc.ccx(0, 1, 3)
    qc.x(0)
    # qc.ccx(0,1,3)
    # qc.ccx(2,3,4)
    # qc.ccx(0,1,3)
    # display(qc.draw('mpl'))
    return qc


def phase_oracle111(n, name='Uf111'):
    qc = QuantumCircuit(n)
    qc.ccx(0, 1, 3)
    qc.ccx(2, 3, 4)
    qc.ccx(0, 1, 3)
    # display(qc.draw('mpl'))
    return qc


def default():
    return "Not a valid number for index search"



def diffuser_function(n, name='V'):
    qc = QuantumCircuit(n, name=name)

    for qb in range(n - 2):  # first layer of Hadamards in diffuser
        qc.h(qb)

    for i in range(n - 2):
        qc.x(i)
    qc.ccx(0, 1, 3)
    qc.ccx(2, 3, 4)
    qc.ccx(0, 1, 3)
    for i in range(n - 2):
        qc.x(i)

    for qb in range(n - 2):  # second layer of Hadamards in diffuser
        qc.h(qb)

    # display(qc.draw('mpl'))
    return qc


n = 5  # n qubits


def Grover_Search(query):
    grc = QuantumCircuit(n, n - 2)
    mu = 1  # number of solutions

    t = int(
        np.floor(np.pi / 4 * np.sqrt(2 ** (n - 2) / mu)))  # Determine r, number of times I have to run the algorithm
    if (query == 'transit == "train"'):
        grc.z(range(n-2))
    else:
        grc.h(range(n - 2))  # step 1: apply Hadamard gates on all working qubits for equal superposition

    # put ancilla in state |-> which is the last qubit
    grc.x(n - 1)
    grc.h(n - 1)

    # step 2: runs t number of loops to get probabilites
    for j in range(t):
        grc.append(phase_oracle111(n),
                   range(n))  ## change phase_oracle function for getting probabilities of other numbers
        grc.append(diffuser_function(n), range(n))
        # grc.compose(phase_oracle(n), range(n))
        # grc.compose(diffuser(n), range(n))
    grc.measure(range(n - 2), range(n - 2))  # step 3: measure all qubits

    # display(grc.draw('mpl'))

    simulator = Aer.get_backend('qasm_simulator')
    job = execute(grc, backend=simulator, shots=8000)
    counts = job.result().get_counts()

    return grc, counts

data_cols = ['latitude', 'longitude',
             'height', 'days_float',
             'rec_date', 'rec_time',
             'timestamp', 'user_id',
             'trip_id', 'transit']
tpoints = (pd.read_csv('010_trip_labeled.csv',
                       index_col=0,
                       names=data_cols,
                       header=0)
             .drop('days_float' ,axis=1)
             .assign(timestamp = lambda x: pd.to_datetime(x.timestamp))
             .sort_values(['trip_id','timestamp'])
          )
tpoints = tpoints.loc[(~tpoints.trip_id.isnull())].copy()
tpoints.head(10)
(tpoints.groupby('trip_id').transit.nunique().max())
tpoints['time_delta'] = (tpoints.timestamp - \
                         tpoints.groupby(['trip_id']).timestamp.shift(1))
tpoints['dt_seconds'] = tpoints['time_delta'].dt.seconds
trip_dt = tpoints.groupby(['trip_id']).dt_seconds.max()
print(trip_dt.mean(), trip_dt.median())
trip_dt.quantile(np.arange(0,1,0.1))
rad_coord = np.radians(tpoints[['latitude','longitude']])
# same DF with the values shifted so that each set of coordinates represents the previous point
# in a given trip for a given index in rad_coord/tpoints
prev_rad_coord = np.radians(tpoints.groupby('trip_id').shift(1)[['latitude','longitude']])
# use haversine formula to compute distance in miles
lat1 = prev_rad_coord['latitude']
lon1 = prev_rad_coord['longitude']
lat2 = rad_coord['latitude']
lon2 = rad_coord['longitude']
dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
# 3965 == radius of the earth in miles - sub 6371 for km distance
delta_dist = 2 * np.arcsin(np.sqrt(a)) * 3956
tpoints['delta_dist'] = delta_dist
tpoints['speed'] = tpoints['delta_dist'] / (tpoints['dt_seconds'] / 3600. )
import time
quantum_start = time.time()
grc, counts = Grover_Search('transit == "train"')
# tpoints.query('transit == "train"').groupby('trip_id').speed.max().describe()
# trip_start = tpoints.groupby('trip_id').timestamp.transform('first')
# trip_end = tpoints.groupby('trip_id').timestamp.transform('last')
# trip_time = trip_end - trip_start
# reg_walks = tpoints.groupby('trip_id').speed.transform('max') <= 15
# min_time = ((trip_time.dt.seconds / 60.) >= 5) & ((trip_time.dt.seconds / 60.) <= 15)
# match_walks = tpoints.loc[reg_walks & min_time,'trip_id'].unique()
# ex_walk = tpoints.trip_id == match_walks[-1]
quantum_end = (time.time() - quantum_start)
plt = plot_histogram(counts)
plt.show()
print("Query time: {0}".format(quantum_end))
# print(len(match_walks))
# print('example trip:', match_walks[-1])
# print('number of GPS observations:', len(tpoints.loc[ex_walk]))
# adj_lat = tpoints.loc[ex_walk,'latitude'] - tpoints.loc[ex_walk,'latitude'].mean()
# adj_lon = tpoints.loc[ex_walk,'longitude'] - tpoints.loc[ex_walk,'longitude'].mean()
# data_plot = pd.concat([(adj_lat * 1000),(adj_lon * 1000)],axis=1).plot(x='longitude',y='latitude',marker='o')
# from matplotlib import pyplot as plt
# plt.boxplot((adj_lat *1000))
# plt.show()

# grc, counts = Grover_Search()

# display(grc.draw('mpl'))
# plt = plot_histogram(counts)
# plt.show()

# In[222]:


max(counts.values())

# In[144]:




