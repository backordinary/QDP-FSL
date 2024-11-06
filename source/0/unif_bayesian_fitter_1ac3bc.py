# https://github.com/bayesian-randomized-benchmarking/qiskit-advocates-bayes-RB/blob/bd9ea7adebf640c6472ef652cc2af53c42ea68d4/unif_bayesian_fitter.py
#Import general libraries (needed for functions)
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

#Import Qiskit classes
import qiskit
from qiskit.tools.monitor import job_monitor
from qiskit import Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error
from qiskit import  QuantumRegister, QuantumCircuit

#Import the RB Functions
import qiskit.ignis.verification.randomized_benchmarking as rb

import copy
import time

# import the bayesian packages
import pymc3 as pm
import arviz as az

from scipy.optimize import curve_fit

def obtain_priors_and_data_from_fitter(rbfit, nCliffs, shots, printout = True):
    
    
    m_gates = copy.deepcopy(nCliffs)
    # We choose the count matrix corresponding to 2 Qubit RB
    Y = (np.array(rbfit._raw_data[0])*shots).astype(int)
    
    # alpha prior and bounds 
    alpha_ref = rbfit._fit[0]['params'][1]    
    #alpha_lower = alpha_ref - 6*rbfit._fit[0]['params_err'][1] 
    #alpha_upper = alpha_ref + 6*rbfit._fit[0]['params_err'][1] 
    alpha_lower = .95*alpha_ref 
    alpha_upper = min(1.05*alpha_ref,1.0) 
    # priors for A anbd B
    mu_AB = np.delete(rbfit._fit[0]['params'],1)
    cov_AB=np.delete(rbfit._fit[0]['params_err'],1)**2
    
    # prior for sigma theta:
    sigma_theta = 0.004 # WIP   
    if printout:
        print("priors:\nalpha_ref",alpha_ref)
        print("alpha_lower", alpha_lower, "alpha_upper", alpha_upper)
        print("A,B", mu_AB, "\ncov A,B", cov_AB)
        print("sigma_theta", sigma_theta)
    
    return m_gates, Y, alpha_ref, alpha_lower, alpha_upper, mu_AB, cov_AB, sigma_theta

# modified for accelerated BM with EPCest as extra parameter
def get_bayesian_model(model_type,Y,shots,m_gates,mu_AB,cov_AB, alpha_ref,
                            alpha_lower=0.5,alpha_upper=0.999,alpha_testval=0.9,                            
                            p_lower=0.9,p_upper=0.999,p_testval=0.95,
                            RvsI=None,IvsR=None,sigma_theta=0.001,
                            sigma_theta_l=0.0005,sigma_theta_u=0.0015): 
# Bayesian model
# from https://iopscience.iop.org/arti=RvsI, cle/1sigma_theta=0.004,0.1088/1367-2630/17/1/013042/pdf 
# see https://docs.pymc.io/api/model.html
    
    RB_model = pm.Model()
    with RB_model:
        # note: shots can be used in place of total_shots (broadcast)
        # to be checked however for all models
        total_shots = np.full(Y.shape, shots) 
       
        #Priors for unknown model parameters
        alpha = pm.Uniform("alpha",lower=alpha_lower,
                           upper=alpha_upper, testval = alpha_ref)
        
        BoundedMvNormal = pm.Bound(pm.MvNormal, lower=0.0)
        
        AB = BoundedMvNormal("AB", mu=mu_AB,testval = mu_AB,
                         cov= np.diag(cov_AB),
                         shape = (2))
        
        if model_type == "hierarchical":
            GSP = AB[0]*alpha**m_gates + AB[1]
            theta = pm.Beta("GSP",
                             mu=GSP,
                             sigma = sigma_theta,
                             shape = Y.shape[1]) 
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts_h", p=theta, observed=Y,
                            n = total_shots)
            
        elif model_type == "h_sigma":
            sigma_t = pm.Uniform("sigma_t",  testval = sigma_theta,
                                upper = sigma_theta_u, lower = sigma_theta_l)
            GSP = AB[0]*alpha**m_gates + AB[1]
            theta = pm.Beta("GSP",
                             mu=GSP,
                             sigma = sigma_t,
                             shape = Y.shape[1])  
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts_h", p=theta, observed=Y,
                            n = total_shots) 
        
        elif model_type == "tilde":
            p_tilde = pm.Uniform("p_tilde",lower=p_lower,
                               upper=p_upper, testval = p_testval)
            GSP = AB[0]*(RvsI*alpha**m_gates + IvsR*(alpha*p_tilde)**m_gates) + AB[1]
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts_t", p=GSP, observed=Y,
                            n = total_shots) 
        
        elif model_type == "h_tilde":
            p_tilde = pm.Uniform("p_tilde",lower=p_lower,
                               upper=p_upper, testval = p_testval)
            sigma_t = pm.Uniform("sigma_t",  testval = sigma_theta,
                                upper = sigma_theta_u, lower = sigma_theta_l)
            
            GSP = AB[0]*(RvsI*alpha**np.tile(m_gates,2) +\
                         IvsR*(alpha*p_tilde)**np.tile(m_gates,2)) \
                         + AB[1] 
            
            theta = pm.Beta("GSP",
                             mu=GSP,
                             sigma = sigma_t,
                             shape = ((2*len(m_gates,))) )                                                
                        
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts_t", p= theta, observed = Y, n = shots) 
        
        else:  # default model "pooled"      
            GSP = AB[0]*alpha**m_gates + AB[1]        
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts_p", p=GSP, observed=Y,
                            n = total_shots) 

    return RB_model

#.reshape( ((Y.shape[1],Y.shape[0])) )
# .reshape( ((len(m_gates),2)) )

# HERE WE USE FOUR UNIFORMS

def build_bayesian_model(model_type,Y,shots,m_gates,mu_AB=None,cov_AB=None, alpha_ref=None,
                            alpha_lower=0.5,alpha_upper=0.999,alpha_testval=0.9, 
                            p_lower=0.9,p_upper=0.999,p_testval=0.95,
                            popt = None, pcov = None,
                            RvsI=None,IvsR=None,sigma_theta=0.001,
                            sigma_theta_l=0.0005,sigma_theta_u=0.0015): 
# Bayesian model
# from https://iopscience.iop.org/arti=RvsI, cle/1sigma_theta=0.004,0.1088/1367-2630/17/1/013042/pdf 
# see https://docs.pymc.io/api/model.html
    
    RB_model = pm.Model()
    with RB_model:
        
        #Priors for unknown model parameters
        
        BoundedUniform = pm.Bound(pm.Uniform,
                                   lower=np.fmax(popt-0.1, np.full(popt.shape,1.e-9)),
                                   upper=np.fmin(popt+0.1, np.full(popt.shape,1.-1e-9)))
        
        if model_type == "hierarchical":
            # legacy: sigma_theta is a scalar
            pi = BoundedUniform("A_α_B", testval = popt, shape = popt.shape) 
                        
            GSP = pi[0]*pi[1]**m_gates + pi[2]
            theta = pm.Beta('θ',
                             mu=GSP,
                             sigma = sigma_theta,
                             shape = Y.shape[1]) 
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts_h", p=theta, observed=Y,
                            n = shots)
            
        elif model_type == "h_sigma":
            pi = BoundedUniform("A_α_B", testval = popt, shape = popt.shape)
                         
            sigma_t = pm.Uniform("σ Beta",  testval = sigma_theta,
                                upper = sigma_theta_u, lower = sigma_theta_l)
            GSP = pi[0]*pi[1]**m_gates + pi[2]
            theta = pm.Beta('θ',
                             mu=GSP,
                             sigma = sigma_t,
                             shape = Y.shape[1])  
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts", p=theta, observed=Y,
                            n = shots) 
        
        elif model_type == "tilde":
            pi = BoundedUniform("A_α_ƥ_B", testval = popt, shape = popt.shape)
                         
            GSP = pi[0]*(RvsI*pi[1]**m_gates + IvsR*(pi[1]*pi[2])**m_gates) + pi[3]
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts_t", p=GSP, observed=Y,
                            n = shots) 
        
        elif model_type == "h_tilde":
            pi = BoundedUniform("A_α_ƥ_B",testval = popt, shape = popt.shape)
                         
            sigma_t = pm.Uniform("σ Beta",  testval = sigma_theta,
                                upper = sigma_theta_u, lower = sigma_theta_l)
            
            GSP = pi[0]*(RvsI*pi[1]**np.tile(m_gates,2) +\
                         IvsR*(pi[1]*pi[2])**np.tile(m_gates,2)) \
                         + pi[3] 
            
            theta = pm.Beta('θ',
                             mu=GSP,
                             sigma = sigma_t,
                             shape = ((2*len(m_gates,))) )                                 
                        
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts", p= theta, observed = Y, n = shots) 
        
        else:  # default model "pooled"
            pi = BoundedUniform("A_α_B", testval = popt, shape = popt.shape)
                         
            GSP = pi[0]*pi[1]**m_gates + pi[2]        
            # Likelihood (sampling distribution) of observations    
            p = pm.Binomial("Counts", p=GSP, observed=Y,
                            n = shots) 

    return RB_model

#.reshape( ((Y.shape[1],Y.shape[0])) )
# .reshape( ((len(m_gates),2)) )


def get_bayesian_model_hierarchical(model_type,Y): # modified for accelerated BM with EPCest as extra parameter
# maintained for backward compatibility, deprecated
# Bayesian model
# from https://iopscience.iop.org/article/10.1088/1367-2630/17/1/013042/pdf 
# see https://docs.pymc.io/api/model.html
    
    RBH_model = pm.Model()
    with RBH_model:
        
        #Priors for unknown model parameters
        alpha = pm.Uniform("alpha",lower=alpha_lower,
                           upper=alpha_upper, testval = alpha_ref)
        
        BoundedMvNormal = pm.Bound(pm.MvNormal, lower=0.0)
        
        AB = BoundedMvNormal("AB", mu=mu_AB,testval = mu_AB,
                         cov= np.diag(cov_AB),
                         shape = (2))

        # Expected value of outcome                
           
        GSP = AB[0]*alpha**m_gates + AB[1]
        
        
        total_shots = np.full(Y.shape, shots)
        theta = pm.Beta("GSP",
                     mu=GSP,
                     sigma = sigma_theta,
                     shape = Y.shape[1])
        
        # Likelihood (sampling distribution) of observations    
        p = pm.Binomial("Counts", p=theta, observed=Y,
                            n = total_shots) 

    return RBH_model

def get_trace(RB_model, draws = 2000, tune= 10000, target_accept=0.95, return_inferencedata=True):
    # Gradient-based sampling methods
    # see also: https://docs.pymc.io/notebooks/sampler-stats.html
    # and https://docs.pymc.io/notebooks/api_quickstart.html
    with RB_model:   
        trace= pm.sample(draws = draws, tune= tune, target_accept=target_accept,
                         return_inferencedata=return_inferencedata)    

    with RB_model:
        az.plot_trace(trace);
        
    return trace

def get_summary(RB_model, trace, round_to=6, hdi_prob=.94, kind='stats'):
    with RB_model:
        #  (hdi_prob=.94 is default)
        az_summary = az.summary(trace, round_to=round_to,  hdi_prob=hdi_prob, kind=kind )  
        
    return az_summary

# obtain EPC from alpha (used by plot_posterior) # deprecated, should use scale
#def alpha_to_EPC(alpha):
        #return 3*(1-alpha)/4

def get_EPC_and_legends(rbfit,azs):
    EPC_Bayes = alpha_to_EPC(azs['mean']['alpha'])
    EPC_Bayes_err = EPC_Bayes - alpha_to_EPC(azs['mean']['alpha']+azs['sd']['alpha'])
    Bayes_legend ="EPC Bayes {0:.5f} ({1:.5f})".format(EPC_Bayes, EPC_Bayes_err)
    Fitter_legend ="EPC Fitter {0:.5f} ({1:.5f})".format(rbfit.fit[0]['epc']\
                                                        ,rbfit._fit[0]['epc_err'])
    if pred_epc > 0.0:
        pred_epc_legend = "EPC predicted {0:.5f}".format(pred_epc)
    else:
        pred_epc_legend = ''
    return EPC_Bayes, EPC_Bayes_err, Bayes_legend,Fitter_legend, pred_epc_legend
    
def EPC_compare_fitter_to_bayes(RB_model, azs, trace,m_name,rbfit):
    EPC_Bayes, EPC_Bayes_err, Bayes_legend,Fitter_legend, pred_epc_legend = get_EPC_and_legends(rbfit,azs)
    with RB_model:
        az.plot_posterior(trace,  var_names=['alpha'], round_to=4,
                          transform = alpha_to_EPC, point_estimate=None)
        plt.title("Error per Clifford  "+RB_process+"  device: "+hardware
                  +'  backend: '+backend.name()+'  model:'+m_name,
                  fontsize=12)
        plt.axvline(x=alpha_to_EPC(alpha_ref),color='red')
        if pred_epc > 0.0:
            plt.axvline(x=pred_epc,color='green') 
            plt.legend((Bayes_legend, "Higher density interval",Fitter_legend, pred_epc_legend), fontsize=10)
        else:
            plt.legend((Bayes_legend, "Higher density interval",Fitter_legend), fontsize=10 )
        
        plt.show()

def GSP_compare_fitter_to_bayes(RB_model, azs,m_name,rbfit):
    EPC_Bayes, EPC_Bayes_err, Bayes_legend,Fitter_legend,_ = get_EPC_and_legends(rbfit,azs)
    # plot ground state population ~ Clifford length
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 6))

    axes.set_ylabel("Ground State Population")
    axes.set_xlabel("Clifford Length")
    axes.plot(m_gates, np.mean(Y/shots,axis=0), 'r.')
    axes.plot(m_gates,azs['mean']['AB[0]']*azs['mean']['alpha']**m_gates+azs['mean']['AB[1]'],'--')
    #axes.plot(m_gates,azs['mean']['GSP'],'--') # WIP
    #axes.errorbar(m_gates, azs['mean']['GSP'], azs['sd']['GSP'], linestyle='None', marker='^') # WIP
    axes.plot(m_gates,mu_AB[0]*np.power(alpha_ref,m_gates)+mu_AB[1],':') 
    for i_seed in range(nseeds):
        plt.scatter(m_gates-0.25, Y[i_seed,:]/shots, label = "data", marker="x")
    axes.legend(["Mean Observed Frequencies",
                 "Bayesian Model\n"+Bayes_legend,
                 "Fitter Model\n"+Fitter_legend],fontsize=12)
    axes.set_title(RB_process+"  device: "+hardware+'  backend: '+backend.name()+'  model:'+m_name,
                   fontsize=14) # WIP

def get_predicted_EPC(error_source):

    #Count the number of single and 2Q gates in the 2Q Cliffords
    gates_per_cliff = rb.rb_utils.gates_per_clifford(transpile_list,xdata[0],basis_gates,rb_opts['rb_pattern'][0])
    for basis_gate in basis_gates:
        print("Number of %s gates per Clifford: %f "%(basis_gate ,
                                                      np.mean([gates_per_cliff[rb_pattern[0][0]][basis_gate],
                                                               gates_per_cliff[rb_pattern[0][1]][basis_gate]])))
    # Calculate the predicted epc
    # from the known depolarizing errors on the simulation
    if error_source == "depolarization":  
        # Error per gate from noise model
        epgs_1q = {'u1': 0, 'u2': p1Q/2, 'u3': 2*p1Q/2}
        epg_2q = p2Q*3/4
        pred_epc = rb.rb_utils.calculate_2q_epc(
            gate_per_cliff=gates_per_cliff,
            epg_2q=epg_2q,
            qubit_pair=[0, 2],
            list_epgs_1q=[epgs_1q, epgs_1q])

    # using the predicted primitive gate errors from the coherence limit
    if error_source == "from_T1_T2": 
        # Predicted primitive gate errors from the coherence limit
        u2_error = rb.rb_utils.coherence_limit(1,[t1],[t2],gate1Q)
        u3_error = rb.rb_utils.coherence_limit(1,[t1],[t2],2*gate1Q)
        epg_2q = rb.rb_utils.coherence_limit(2,[t1,t1],[t2,t2],gate2Q)
        epgs_1q = {'u1': 0, 'u2': u2_error, 'u3': u3_error}
        pred_epc = rb.rb_utils.calculate_2q_epc(
            gate_per_cliff=gates_per_cliff,
            epg_2q=epg_2q,
            qubit_pair=[0, 1],
            list_epgs_1q=[epgs_1q, epgs_1q])
    return pred_epc

def get_and_run_seeds(rb_circs, shots, backend, coupling_map,
                      basis_gates, noise_model, retrieve_list=[]):   
    #basis_gates = ['u1','u2','u3','cx'] # use U,CX for now
    result_list = []
    transpile_list = []

    for rb_seed,rb_circ_seed in enumerate(rb_circs):
        print('Compiling seed %d'%rb_seed)
        rb_circ_transpile = qiskit.transpile(rb_circ_seed,
                                             optimization_level=0,
                                             basis_gates=basis_gates)
        print('Runing seed %d'%rb_seed)

        if retrieve_list == []:
            if noise_model == None: # this indicates harware run          
                job = qiskit.execute(rb_circ_transpile, 
                                 shots=shots,
                                 backend=backend,
                                 coupling_map=coupling_map,
                                 basis_gates=basis_gates)
            else:
                job = qiskit.execute(rb_circ_transpile, 
                                 shots=shots,
                                 backend=backend,
                                 coupling_map=coupling_map,
                                 noise_model=noise_model,
                                 basis_gates=basis_gates)                        
            job_monitor(job)
        else: 
            job = backend.retrieve_job(retrieve_list[rb_seed])

        result_list.append(job.result())
        transpile_list.append(rb_circ_transpile)    

    print("Finished  Jobs")
    return result_list, transpile_list

def get_count_data(result_list, nCliffs):
### another way to obtain the observed counts
#corrected for accomodation pooled data from 1Q, 2Q and 3Q interleave processes
    list_bitstring = ['0','00', '000', '100'] # all valid bistrings
    Y_list = []
    for rbseed, result in enumerate(result_list):
        row_list = []
        for c_index, c_value in enumerate(nCliffs) :  
            total_counts = 0
            for key,val in result.get_counts()[c_index].items():
                if  key in list_bitstring:
                    total_counts += val
                    #print(key,val,total_counts)
            row_list.append(total_counts)
        Y_list.append(row_list)    
    return np.array(Y_list)

# This section for the LS fit in this model pooling
# data from 2Q and 3Q interleave processes

def func(x, a, b, c):
    return a * b ** x + c

def epc_fitter_when_mixed_2Q_3Q_RB(X,Y1,Y2,shots,check_plot=False):

    xdata = np.array(list(X)*Y1.shape[0]) # must be something simpler
    ydata1 = np.ravel(Y1)/shots     
    popt, pcov = curve_fit(func, xdata, ydata1)
    perr= np.sqrt(np.diag(pcov))
    
    ydata2 = np.ravel(Y2)/shots  
    popt2, pcov2 = curve_fit(func, xdata, ydata2)
    perr2= np.sqrt(np.diag(pcov2))
    
    if check_plot:
        import matplotlib.pyplot as plt
        plt.plot(xdata, ydata1, 'bx', label='Reference')
        plt.plot(xdata, ydata2, 'r+', label='Interleave')
        plt.plot(X, np.mean(Y1,axis=0)/shots, 'b-', label=None)
        plt.plot(X, np.mean(Y2,axis=0)/shots, 'r-', label=None)
        plt.ylabel('Population of |00>')
        plt.xlabel('Number of Cliffords')
        plt.legend()
        plt.show()
        
        print(popt[1])
        print(perr[1])
        print(popt2[1])
        print(perr2[1])

    epc_est_fitter = 3*(1 - popt2[1]/popt[1])/4
    epc_est_fitter_err = 3*(popt2[1]/popt[1])/4 * (np.sqrt(perr[1]**2 + perr2[1]**2))
    return epc_est_fitter, epc_est_fitter_err

# This section for the demo with qiskit experiment


def retrieve_from_lsf(exp):
    perr_fm = np.sqrt(np.diag(exp._analysis_results[0]['pcov']))
    popt_fm = exp._analysis_results[0]['popt']
    epc_est_fm = exp._analysis_results[0]['EPC']
    epc_est_fm_err = exp._analysis_results[0]['EPC_err']
    experiment_type = exp._data[0]['metadata']['experiment_type']
    
    return perr_fm, popt_fm, epc_est_fm, epc_est_fm_err, experiment_type

def get_GSP_counts(data, x_length, data_range):
#obtain the observed counts used in the bayesian model
#corrected for accomodation pooled data from 1Q, 2Q and 3Q interleave processes
    list_bitstring = ['0','00', '000', '100'] # all valid bistrings
    Y_list = []
    for i_samples in data_range:
        row_list = []
        for c_index in range(x_length) :  
            total_counts = 0
            i_data = i_samples*x_length + c_index
            for key,val in data[i_data]['counts'].items():
                if  key in list_bitstring:
                    total_counts += val
            row_list.append(total_counts)
        Y_list.append(row_list)    
    return np.array(Y_list)

def RB_bayesian_results(resmodel, trace, lengths,
                       epc_est_fm, epc_est_fm_err, experiment_type, scale,                                        num_samples, Y, shots, physical_qubits,
                       interleaved_gate, backend,
                       EPG_dic = None, epc_calib = np.nan,
                       Y1 = None, Y2= None, show_plot = True,
                       routine = 'get_bayesian_model'):
    
    # obtain EPC from alpha (used by az.plot_posterior) 
    def alpha_to_EPC(alpha):
        return scale*(1-alpha)
    
    azt_summary = get_summary(resmodel, trace, kind = 'stats')
                              
    print(azt_summary,'\n')
    
    if experiment_type == "StandardRB":
        if routine == 'get_bayesian_model':
            v_name = 'alpha'
            alpha = azt_summary['mean']['alpha']
            sd_alpha = azt_summary['sd']['alpha']
            A = azt_summary['mean']['AB[0]']
            B = azt_summary['mean']['AB[1]']
        elif routine == 'build_bayesian_model':
            v_name = 'A_α_B'
            alpha = azt_summary['mean']['A_α_B[1]']
            sd_alpha = azt_summary['sd']['A_α_B[1]']
            A = azt_summary['mean']['A_α_B[0]']
            B = azt_summary['mean']['A_α_B[2]']
        epc_est_a = scale*(1 - alpha)
        epc_est_a_err = scale * sd_alpha

        # compare LSF and SMC
        print("Model:    Frequentist        Bayesian")
        print("_______________________________________")
        print("EPC         {0:1.3e}        {1:1.3e} "
              .format(epc_est_fm,epc_est_a))  
        print("± sigma    ± {0:1.3e}      ± {1:1.3e} "
              .format(epc_est_fm_err, epc_est_a_err))
        
        for i, (gate,EPG) in enumerate(EPG_dic.items()):
            print("{0:<12}{1:1.3e}        {2:1.3e}"
              .format("EPG "+gate,EPG,EPG*epc_est_a/epc_est_fm)) 

        if show_plot == False:
            return
        
        import matplotlib.pyplot as plt # seems we need to reimport for replot WIP
        fig, plt = plt.subplots(1, 1)

        plt.set_ylabel("P(0)")
        plt.set_xlabel("Cliffords Length")
        
        plt.plot(lengths,A*alpha**lengths + B,'-',color="r")
        for i_seed in range(num_samples):
            plt.scatter(lengths, Y[i_seed,:]/shots, label = "data", marker="x",color="grey")
        plt.set_title(experiment_type +', ' + "qubit: "  + str(physical_qubits)\
                         +', backend: '+backend.name(),
                        fontsize=14);
             
    elif experiment_type == "InterleavedRB": 
        
        if routine == 'get_bayesian_model':
            v_name = 'p_tilde'
            alpha = azt_summary['mean']['alpha']
            p = azt_summary['mean']['p_tilde']
            sd_p = p = azt_summary['sd']['p_tilde']
            A = azt_summary['mean']['AB[0]']
            B = azt_summary['mean']['AB[1]']
        elif routine == 'build_bayesian_model':
            v_name = 'A_α_ƥ_B'
            alpha = azt_summary['mean']['A_α_ƥ_B[1]']
            p = azt_summary['mean']['A_α_ƥ_B[2]']
            sd_p = azt_summary['sd']['A_α_ƥ_B[2]']
            A = azt_summary['mean']['A_α_ƥ_B[0]']
            B= azt_summary['mean']['A_α_ƥ_B[3]']
        epc_est_a = scale*(1 - p)
        epc_est_a_err = scale * sd_p

        # compare LSF and SMC
        print("Model:    Frequentist         Bayesian        Calibration")
        print("__________________________________________________________")
        print("EPC         {0:1.3e}        {1:1.3e}         {2:1.3e}"
              .format(epc_est_fm,epc_est_a,epc_calib ))  
        print("± sigma    ± {0:1.3e}      ± {1:1.3e} "
              .format(epc_est_fm_err, epc_est_a_err))
        
        if show_plot ==False:
            return

        import matplotlib.pyplot as plt # seems we need to reimport for replot WIP

        fig, plt = plt.subplots(1, 1)

        plt.set_ylabel("P(0)")
        plt.set_xlabel("Cliffords Length")


        for i_seed in range(num_samples):
            plt.scatter(lengths, Y1[i_seed,:]/shots, label = "data", marker="x",color="r")
            plt.scatter(lengths, Y2[i_seed,:]/shots, label = "data", marker="+",color="orange")
        plt.plot(lengths,A*alpha**lengths + B,'--',color="r")
        plt.plot(lengths,A*(alpha*p)**lengths + B,'--',color="orange")

        plt.legend(("Standard, SMC model",
                     "Interleaved, SMC model"))

        plt.set_title(experiment_type +', ' + interleaved_gate  + str(physical_qubits)\
                         +', backend: '+backend.name(),
                        fontsize=14);

    import matplotlib.pyplot as plt # if not yet imported 
    #plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"] 
    # to reset to default
    plt.rcParams["figure.figsize"] = (8,5)
    
    if routine == 'build_bayesian_model': # WIP
        return
    
    with resmodel:
            ax = az.plot_posterior(trace,  var_names=[v_name],
                                   round_to=4, point_estimate=None, 
                                   transform =  alpha_to_EPC)
            ax.set_xlim(epc_est_a - 6*epc_est_a_err, epc_est_a + 6*epc_est_a_err)
            plt.axvline(x=epc_est_fm,color='cyan',ls="-")    
            if epc_calib != np.nan:
                plt.axvline(x=epc_calib,color='r',ls=":")
            plt.axvline(x=epc_est_a,color='blue',ls=":")
            plt.title(experiment_type +', ' + interleaved_gate  + " qubit(s):" + str(physical_qubits)\
                     +', backend: '+backend.name(),
                    fontsize=14)
            Bayes_legend =  "EPC  SMC: {0:1.3e} ± {1:1.3e}".format(epc_est_a, epc_est_a_err)
            LSF_legend =    "EPC  LSF: {0:1.3e} ± {1:1.3e}".format(epc_est_fm, epc_est_fm_err)  
            Cal_legend =    "EPC Calibration: {0:1.3e}".format(epc_calib)
            if epc_calib > 0.0:   
                plt.legend((Bayes_legend, "$Highest\; density\; interval$ HDI",
                            LSF_legend,
                            Cal_legend), fontsize=12 )
            else: 
                plt.legend((Bayes_legend, "$Highest\; density\; interval$ HDI",
                            LSF_legend), fontsize=12 )
                
# obtain EPC from alpha and scale(used by az.plot_posterior) 
def alpha_to_EPC_from_scale(alpha, scale):
    return scale*(1-alpha)

# guess number of shots
def guess_shots(Y):
    shot_exp = 1
    test_shot = np.max(Y)
    while test_shot > 2**shot_exp:
           shot_exp += 1
    return 2**shot_exp                     
               
def bayesian_standard_RB_model(): 
    # construct model
    RB_model = get_bayesian_model(model_type="pooled",Y=Y,shots=shots,m_gates=lengths,
                                           mu_AB=[popt_fm[0],popt_fm[2]],cov_AB=[perr_fm[0],perr_fm[2]],
                                           alpha_ref=popt_fm[1],
                                           alpha_lower=popt_fm[1]-6*perr_fm[1],
                                           alpha_upper=min(1.-1.E-6,popt_fm[1]+6*perr_fm[1]),
                                           RvsI=None,IvsR=None)
    return RB_model

def bayesian_interleaved_RB_model(): 
    # construct model
    RB_model =  get_bayesian_model("tilde",Y=Y,shots=shots, m_gates=lengths,
                                         alpha_ref=popt_fm[1], p_testval= popt_fm[2],
                                         alpha_lower=popt_fm[1]-6*perr_fm[1],
                                         alpha_upper=min(1.-1.E-6,popt_fm[1]+6*perr_fm[1]),                            
                                         p_lower=popt_fm[2]-6*perr_fm[2],
                                         p_upper=min(1.-1.E-6,popt_fm[2]+6*perr_fm[2]),
                                         mu_AB=[popt_fm[0],popt_fm[3]],cov_AB=[perr_fm[0],perr_fm[3]],
                                         RvsI=RvsI,IvsR=IvsR)
    return RB_model

