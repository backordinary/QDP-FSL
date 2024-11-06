# https://github.com/ZevVladimir/FIRE_Project/blob/9060d5bb592709984f71fadfc28da8bb03ec4981/qiskit_neural_network.py
import numpy as np
import matplotlib.pyplot as plt

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.visualization import *

#These imports were used within the IBM environment and don't work here but aren't necessary
#from qiskit.tools.jupyter import *
#from ibm_quantum_widgets import *
#from qiskit.providers.aer import QasmSimulator

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy import stats
from scipy import special
from scipy import interpolate
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from matplotlib import pyplot as plt
from IPython.display import clear_output
import time
from qiskit_machine_learning.algorithms import VQC

#path to data
file_path = "/home/zeevvladimir/Personal_Project/TNG300_RF_data-20221026T024254Z-001/TNG300_RF_data/"

# this function takes the number of galaxies per halo (counts), masses of each halo that has at least one galaxy (masses) and masses of all host halos
# (which is to say big halos) regardless of whether or not they host one of our 8000 galaxies (all_masses). It then returns the HOD as well as the bin centers

def get_hod(masses, counts, all_masses, std):
    # number of bin edges
    n_bins = 31
   
    # bin edges (btw Delta = np.log10(bins[1])-np.log10(bins[0]) in log space)
    bins = np.logspace(10.,15.,n_bins)
   
    # bin centers
    bin_cents = 0.5*(bins[1:]+bins[:-1])
    
    if std == True:
        # This histogram tells you the standard deviation of halo masses in each bin interval (n_bins-1)
        # This statstic function takes: variable to use for binning, variable to do statistics on
        # so here we are binning according to masses and taking the standard deviation of the galaxy counts
        hist_hod, edges, bin_number  = stats.binned_statistic(masses, counts, 'std', bins=bins)
   
    elif std == False:
        # This histogram tells you how many halos there are in each bin inerval (n_bins-1)
        hist_norm, edges = np.histogram(all_masses,bins=bins)
        hist_weighted, edges = np.histogram(masses,bins=bins,weights=counts)
        hist_hod = hist_weighted/hist_norm # to get average number of halo living in each bin
    

    return hist_hod, bin_cents


def get_hist_count(inds_top,sub_parent_id, sub_pos, group_mass, group_pos, N_halos, std):
    
    # Find the halo ID's of the galaxies' halo parents in the original group_mass array
    hosts_top = sub_parent_id[inds_top]

    # find the positions of the galaxies (i.e. the subhalos that we have decided to call "galaxies")
    pos_top = sub_pos[inds_top]
    
    # Find the masses of their halo parents from the original group_mass array
    masses_top = group_mass[hosts_top]
    
    # Knowing the ID's of the relevant halos (i.e. those who are hosting a galaxy),
    # tell me which ID's are unique, what indices in the hosts_top array these
    # unique ID's correspond to and how many times they each repeat
    hosts, inds, gal_counts = np.unique(hosts_top,return_index=True,return_inverse=False, return_counts=True)

    # get unique masses of hosts to compute the HOD (histogram of average occupation number as a function of mass)
    host_masses = masses_top[inds]
 
    hist, bin_cents = get_hod(host_masses,gal_counts,group_mass, std)

    # get the galaxy counts per halo
    count_halo = np.zeros(N_halos,dtype=int)
    count_halo[hosts] += gal_counts
    
    return hist, bin_cents, count_halo

Group_M_Mean200_dm = np.load(file_path + 'new_Group_M_Mean200_dm.npy')
GroupPos_dm = np.load(file_path + 'new_GroupPos_dm.npy')
GroupConc_dm = np.load(file_path + 'new_GroupConc_dm.npy')
GroupEnv_dm = np.load(file_path + 'new_GroupEnv_dm.npy')
GroupEnvAnn_dm = np.load(file_path + 'new_GroupEnvAnn_dm.npy')
GroupEnvTH_dm = np.load(file_path + 'new_GroupEnvTH_dm.npy') #already masked for Mhalo>1e11
GroupSpin_dm = np.load(file_path + 'new_GroupSpin_dm.npy')
GroupNsubs_dm = np.load(file_path + 'new_GroupNsubs_dm.npy')
GroupVmaxRad_dm = np.load(file_path + 'new_GroupVmaxRad_dm.npy')
Group_SubID_dm = np.load(file_path + 'new_Group_SubID_dm.npy') #suhalo ID's
Group_Shear_dm = np.load(file_path + 'new_Group_Shear_dm.npy') #already,masked for Mhalo>1e11
SubVdisp_dm = np.load(file_path + 'new_SubVdisp_dm.npy')
SubVmax_dm = np.load(file_path + 'new_SubVmax_dm.npy')
SubGrNr_dm = np.load(file_path + 'new_SubGrNr_dm.npy') #Index into the Group table of the FOF host/parent of Subhalo
SubhaloPos_dm = np.load(file_path + 'new_SubhaloPos_dm.npy')
count_dm = np.load(file_path + 'new_count_dm.npy')
cent_count_dm = np.load(file_path + 'new_cent_count_dm.npy')
sat_count_dm = count_dm-cent_count_dm
GroupEnvTH_1_3 = np.load(file_path + 'new_GroupEnvTH_1_3.npy') #env at 1.3 Mpc
GroupEnvTH_2_5 = np.load(file_path + 'new_GroupEnvTH_2_5.npy')#env at 2.6 Mpc

#mask off the lowest mass halos
mass_mask = Group_M_Mean200_dm>1e11

#Interpolate the shear at the radius of the halo using group shear file
rEnv=np.logspace(np.log10(0.4),np.log10(10),20) #scales at which shear was calculated
rad=(np.load(file_path + 'new_Group_R_Mean200_dm.npy')/1e3)[mass_mask] #halo radius
shear=np.zeros(len(rad))

for i in range(len(rad)):
    ShearFit=interpolate.InterpolatedUnivariateSpline(rEnv,Group_Shear_dm[i])
    shear[i]=ShearFit(1*rad[i])
    mask = shear<0; shear[mask]=1e-3 #To ameliorate unrealistic values
shear[mask].shape
rEnv

shear_1Mpc = Group_Shear_dm[:,7] # we also want shear value at approx at 1.3Mpc

#create testing cube
boxLen=137
maskBox=GroupPos_dm[mass_mask][:,0]<boxLen
maskBox*=GroupPos_dm[mass_mask][:,1]<boxLen
maskBox*=GroupPos_dm[mass_mask][:,2]<boxLen

#organize data for training/testing
#features
mass_train = Group_M_Mean200_dm[mass_mask][~maskBox]
mass_test = Group_M_Mean200_dm[mass_mask][maskBox]
env_train = GroupEnv_dm[mass_mask][~maskBox]
env_test = GroupEnv_dm[mass_mask][maskBox]
envann_train = GroupEnvAnn_dm[mass_mask][~maskBox]
envann_test = GroupEnvAnn_dm[mass_mask][maskBox]
envth_train = GroupEnvTH_dm[~maskBox]
envth_test = GroupEnvTH_dm[maskBox]
conc_train = GroupConc_dm[mass_mask][~maskBox]
conc_test = GroupConc_dm[mass_mask][maskBox]
spin_train = GroupSpin_dm[mass_mask][~maskBox]
spin_test = GroupSpin_dm[mass_mask][maskBox]
ngals_train = GroupNsubs_dm[mass_mask][~maskBox]
ngals_test = GroupNsubs_dm[mass_mask][maskBox]
#vdisp_train = parents_Vdisp[mass_mask][~maskBox]
#vdisp_test = parents_Vdisp[mass_mask][maskBox]
#vmax_train = parents_Vmax[mass_mask][~maskBox]
#vmax_test = parents_Vmax[mass_mask][maskBox]
vmax_rad_train = GroupVmaxRad_dm[mass_mask][~maskBox]
vmax_rad_test = GroupVmaxRad_dm[mass_mask][maskBox]
shear_train = shear[~maskBox]
shear_test = shear[maskBox]
shear_1Mpc_train = shear_1Mpc[~maskBox]
shear_1Mpc_test = shear_1Mpc[maskBox]
envth_1Mpc_train = GroupEnvTH_1_3[~maskBox]
envth_1Mpc_test = GroupEnvTH_1_3[maskBox]
envth_2Mpc_train = GroupEnvTH_2_5[~maskBox]
envth_2Mpc_test = GroupEnvTH_2_5[maskBox]
#labels
#number of galaxy counts
counts_train = count_dm[mass_mask][~maskBox]
counts_test = count_dm[mass_mask][maskBox]
#number of satellite counts
sat_counts_train = sat_count_dm[mass_mask][~maskBox]
sat_counts_test = sat_count_dm[mass_mask][maskBox]
#number of central counts
cent_counts_train = cent_count_dm[mass_mask][~maskBox]
cent_counts_test = cent_count_dm[mass_mask][maskBox]

## make arrays holding all the parameters
n_params=12
train_params = np.zeros((mass_train.shape[0],n_params), dtype = np.float64)
train_params[:,0] = mass_train
train_params[:,1] = envann_train
train_params[:,2] = envth_train
train_params[:,3] = envth_1Mpc_train
train_params[:,4] = envth_2Mpc_train
train_params[:,5] = env_train #GS
train_params[:,6] = conc_train
train_params[:,7] = shear_train
train_params[:,8] = shear_1Mpc_train
train_params[:,9] = spin_train
#train_params[:,10] = vmax_train
#train_params[:,11] = vdisp_train

test_params = np.zeros((mass_test.shape[0],n_params), dtype = np.float64)
test_params[:,0] = mass_test
test_params[:,1] = envann_test
test_params[:,2] = envth_test
test_params[:,3] = envth_1Mpc_test
test_params[:,4] = envth_2Mpc_test
test_params[:,5] = env_test
test_params[:,6] = conc_test
test_params[:,7] = shear_test
test_params[:,8] = shear_1Mpc_test
test_params[:,9] = spin_test
#test_params[:,10] = vmax_test
#test_params[:,11] = vdisp_test

# choose your paramter for training and testing
param_indices = [0,5,7,9]
X_train= train_params[:,param_indices]
X_test = test_params[:,param_indices]
print("Train shape: " + str(X_train.shape))
print(X_train)
print("Test shape: " + str(X_test.shape))

y_train = counts_train
y_test = counts_test

def HODCent(mH,mMin,sigma):
    out=1/2*(1.+special.erf((np.log10(mH)-mMin)/sigma))
    #out=1/2*(1.+special.erf(((mH)-mMin)/sigma))
    return(out)

def HODSat(mH,mCut,m1,alpha):
    mask=mH<=10**mCut
    out=((mH-10**mCut)/(10**m1))**alpha
    out[mask]=0.
    return(out)

# get the hod from TNG counts using imported function
std = False #we use std=True when we want the scatter within the bins
hod_true_sat, bin_cents = get_hod(mass_test,sat_counts_test,mass_test, std)
hod_true_cent, bin_cents = get_hod(mass_test,cent_counts_test,mass_test, std)
hod_true_tot, bin_cents = get_hod(mass_test,counts_test,mass_test, std)

nan_mask = np.logical_not(np.isnan(hod_true_sat))

#fit the HODsat function
sat_params, cov = curve_fit(HODSat, bin_cents[nan_mask],hod_true_sat[nan_mask], p0=[12.5,13.5,0.9])
sat_params

#adjust by hand if needed
mH = mass_test
mCut = 12.71
m1 = 13.65
alpha = 1  

fit_sat_counts = HODSat(mH,mCut,m1,alpha)

# adjust by hand if needed:
sigma = 0.23 #np.round(cent_params[1],2)
mMin = 12.7 #np.round(cent_params[0],2)

#fit the HODcent function
cent_params, cov = curve_fit(HODCent, bin_cents[nan_mask],hod_true_cent[nan_mask], p0=[0.5,12.5])
cent_params

fit_cent_counts = HODCent(mass_test,mMin,sigma)
fit_tot_counts = fit_cent_counts + fit_sat_counts

#get fitted HODs
hod_fit, bin_cents = get_hod(mass_test, fit_tot_counts, mass_test, std)
hod_fit_sat, bin_cents = get_hod(mass_test, fit_sat_counts, mass_test, std)
hod_fit_cent, bin_cents = get_hod(mass_test, fit_cent_counts, mass_test, std)

# plt.figure(figsize=(8,6))
# plt.plot(bin_cents, hod_true_tot,label='Total Subbox TNG Counts')
# plt.plot(bin_cents, hod_fit, linestyle = '--',label='HOD fit Total Counts')
# plt.plot(bin_cents, hod_true_sat, label='TNG contribution from satellites')
# plt.plot(bin_cents, hod_fit_sat, linestyle = '--',label='HOD fit sats')
# plt.plot(bin_cents, hod_true_cent, label='TNG contribution from centrals')
# plt.plot(bin_cents, hod_fit_cent, linestyle = '--',label='HOD fit centrals')
# plt.yscale('log')
# plt.xscale('log')
# plt.ylabel(r'$\langle N_\mathrm{gals} \rangle$')
# plt.xlabel(r'$M\ [h^{-1}M_\mathrm{halo}]$')
# plt.xlim(5e11,1e15)
# plt.ylim(0.01,30)
# plt.legend(fontsize='x-small')


#determine feature map, ansatz, and optimizer
num_features = len(param_indices) 

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", fold=20)

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=100)

#TODO replace with sampler
quantum_instance = QuantumInstance(
    AerSimulator(),
    shots=1024,
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)

objective_func_vals = []
#plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.draw()

#choose a random amount of indices to use to train the model
random_indices_train = np.random.randint(0,100000, (500))
X_train = X_train[random_indices_train]
y_train = y_train[random_indices_train]

#create the VQC model 
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    quantum_instance=quantum_instance,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

#train the model
start = time.time()
print("Starting")
vqc.fit(X_train, y_train)
elapsed = time.time() - start
print(f"Training time: {round(elapsed)} seconds")
plt.show()


random_indices_test = np.random.randint(0,80000, (500))
X_test = X_test[random_indices_test]
y_test = y_test[random_indices_test]
fit_tot_counts = fit_tot_counts[random_indices_test]

train_score_q4 = vqc.score(X_train, y_train)
test_score_q4 = vqc.score(X_test, y_test)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

ypred = vqc.predict(X_test)

#graph the predictions of VQC for the number of galaxies
x = np.linspace(np.min(y_test),np.max(y_test), 100)
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(y_test,ypred, label = '$\mathrm{VQC}$')
ax.scatter(y_test,fit_tot_counts, label = '$\mathrm{HOD}$')
ax.plot(x, x, linestyle='-', c='r')
ax.set_xlabel(r'$\rm{TNG300}\ N_{\rm gals} $', fontsize = 20)
ax.set_ylabel(r'$\rm{PREDICTED}\ N_{\rm gals} $', fontsize = 20)
ax.set_title(r'$\rm{Prediction\ results\ |\ Subbox\ TNG300}$')
plt.legend()