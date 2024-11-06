# https://github.com/LucianoPereiraValenzuela/Parallel_QND_measurement_tomography/blob/913a2ee9b393338066da6f16ecf2b0b8452bf159/codes/.ipynb_checkpoints/main_v2-checkpoint.py
import numpy as np
import matplotlib.pyplot as plt
import QuantumTomography as qt
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.visualization import plot_gate_map
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import seaborn as sns
import networkx as nx
color_map = sns.cubehelix_palette(reverse=True, as_cmap=True)
from joblib import Parallel, delayed


def get_noise( job ):
    readout_error = [ job.properties().readout_error(j) for j in range(7)  ]
    T1 = [ job.properties().t1(j) for j in range(7)  ]
    return readout_error, T1

def plot_error_map( backend, single_gate_errors, double_gate_errors ):

    single_gate_errors = 100*single_gate_errors
    single_norm = matplotlib.colors.Normalize( vmin=min(single_gate_errors), vmax=max(single_gate_errors))
    q_colors = [color_map(single_norm(err)) for err in single_gate_errors]
    
    double_gate_errors = 100*double_gate_errors
    double_norm = matplotlib.colors.Normalize( vmin=min(double_gate_errors), vmax=max(double_gate_errors))
    l_colors = [color_map(double_norm(err)) for err in double_gate_errors]
    
    figsize=(12, 9)
    fig = plt.figure(figsize=figsize)
    gridspec.GridSpec(nrows=2, ncols=3)

    grid_spec = gridspec.GridSpec(
        12, 12, height_ratios=[1] * 11 + [0.5], width_ratios=[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    )

    left_ax = plt.subplot(grid_spec[2:10, :1])
    main_ax = plt.subplot(grid_spec[:11, 1:11])
    right_ax = plt.subplot(grid_spec[2:10, 11:])
    bleft_ax = plt.subplot(grid_spec[-1, :5])
    bright_ax = plt.subplot(grid_spec[-1, 7:])

    plot_gate_map(backend, qubit_color=q_colors, line_color=l_colors, line_width=5,
                plot_directed=False,
                ax=main_ax )

    main_ax.axis("off")
    main_ax.set_aspect(1)

    single_cb = matplotlib.colorbar.ColorbarBase(
                bleft_ax, cmap=color_map, norm=single_norm, orientation="horizontal"
            )
    tick_locator = ticker.MaxNLocator(nbins=5)
    single_cb.locator = tick_locator
    single_cb.update_ticks()
    single_cb.update_ticks()
    bleft_ax.set_title(f"H error rate")

    cx_cb = matplotlib.colorbar.ColorbarBase(
                bright_ax, cmap=color_map, norm=double_norm, orientation="horizontal"
            )
    tick_locator = ticker.MaxNLocator(nbins=5)
    cx_cb.locator = tick_locator
    cx_cb.update_ticks()
    bright_ax.set_title(f"CNOT error rate")
    
    return fig

def get_backend_conectivity(backend):
	"""
	Get the connected qubit of a backend. Has to be a quantum computer.

	Parameters
	----------
	backend: qiskit.backend

	Return
	------
	connexions: (list)
		List with the connected qubits
	"""
	defaults = backend.defaults()
	connexions = [indx for indx in defaults.instruction_schedule_map.qubits_with_instruction('cx')]
	return connexions


def marginal_counts_dictionary( counts , idx ):
    
    if len(idx) == 0 :
        marginal_counts = counts
    else:
        marginal_counts = {}
        for key in counts:
            sub_key = ''
            for k in idx:
                sub_key += key[k]
            if sub_key in marginal_counts:
                marginal_counts[sub_key] += counts[key]
            else:
                marginal_counts[sub_key] = counts[key]
                
    return marginal_counts


def dict2array(counts, n_qubits ):

    p = np.zeros( 2**n_qubits )

    for idx in counts :
        p[ int(idx[::-1],2) ] = counts[idx]
               
    return p.reshape( n_qubits*[2] )


def resampling_counts( counts, resampling=0 ):
    
    if resampling > 0 :
        keys  = counts.keys()
        probs = np.array(list(counts.values()))
        probs = np.random.multinomial( resampling, probs/np.sum(probs) )
        counts = dict(zip(keys, probs) )
    
    return counts


def tomographic_gate_set(n=1):
    """
    Create circuits for perform Pauli tomography of a single qubit.
    """
    
    circ_0 = QuantumCircuit(n)

    circ_x = QuantumCircuit(n)
    circ_x.x(range(n))

    circ_h = QuantumCircuit(n)
    circ_h.h(range(n))

    circ_k = QuantumCircuit(n)
    circ_k.u( np.pi/2, np.pi/2, -np.pi/2, range(n))

    circ_gates = [ circ_0, circ_x, circ_h, circ_k]

    return circ_gates


class tomographic_gate_set_tomography:
    
    def __init__( self, n ):
        self._n = n
        
    def circuits( self ):
        n = self._n
        circ_gates = tomographic_gate_set(n)
        circ_gst = [] 

        for circ_j in circ_gates :
            for circ_i in circ_gates :
                for circ_k in circ_gates :
                    qc = QuantumCircuit(n)
                    qc.compose( circ_i, range(n), inplace=True )
                    qc.compose( circ_j, range(n), inplace=True )
                    qc.compose( circ_k, range(n), inplace=True )
                    qc.measure_all()
                    circ_gst.append( qc )
        
        self._circ_gst = circ_gst

        return circ_gst        
        
    def fit( self, results, circ_gst=None, resampling=0 ):
        
        if circ_gst is None :
            circ_gst = self._circ_gst
        
        self._counts = []
        for qc in circ_gst:
            counts  = resampling_counts( results.get_counts(qc), resampling=resampling )
            self._counts.append( counts )
        
        rho = np.array([1,0,0,0])
        Detector = np.array([ [1,0], [0,0], [0,0], [0,1] ])
        I = np.kron( np.eye(2), np.eye(2) )
        X = np.kron( qt.PauliMatrices(1), qt.PauliMatrices(1) )
        H = np.kron( qt.PauliMatrices(1) + qt.PauliMatrices(3), qt.PauliMatrices(1) + qt.PauliMatrices(3) )/2
        K = np.kron( qt.PauliMatrices(0) + 1j*qt.PauliMatrices(1), qt.PauliMatrices(0) - 1j*qt.PauliMatrices(1) )/2
        Gates = np.array([I,X,H,K])

        rho_hat_all    = []
        Detetor_hat_all = []
        Gates_hat_all   = []
        for m in range(self._n) :
            probs = []
            for counts in self._counts:
                probs_temp = dict2array( marginal_counts_dictionary( counts, [m] ), 1 ) 
                probs.append( probs_temp/np.sum(probs_temp)  )
            del probs_temp
            probs = np.array( probs ).reshape(4,4,4,2)
            rho_hat, Detetor_hat, Gates_hat = qt.MaximumLikelihoodGateSetTomography( probs, rho, 
                                                                                    Detector, Gates, 'detector')
            rho_hat_all.append( rho_hat )
            Detetor_hat_all.append( Detetor_hat )
            Gates_hat_all.append( Gates_hat )
        
        return [rho_hat_all, Detetor_hat_all, Gates_hat_all]
            
            
class measurement_process_tomography:           
            
    def __init__( self, n=1, p=None ):
        
        self._n = n
        self._p = p
        
        
    def circuits(self):
        
        n = self._n
        p = self._p
        
        circ_0, circ_x, circ_h, circ_k = tomographic_gate_set(p)
        
        if n == 1 :
            
            circs_mpt = []
            for circ_hk in [circ_0, circ_h, circ_k ]:
                for circ_0x in  [ circ_0, circ_x ]:
                    for circ_xyz in [circ_0, circ_h, circ_k ]:
                        qc = QuantumCircuit(p,2*p)
                        qc.compose( circ_0x, range(p), inplace=True )
                        qc.compose( circ_hk, range(p), inplace=True )
                        qc.barrier()
                        qc.measure( range(p), range(p) )
                        qc.compose( circ_xyz, range(p), inplace=True )
                        qc.barrier()
                        qc.measure( range(p), range(p,2*p)  )
                        circs_mpt.append( qc )
            
        else :
            circs_state_s = []
            for circ_hk in [circ_0, circ_h, circ_k ]:
                for circ_0x in  [ circ_0, circ_x ]:
                    qc = QuantumCircuit(p)
                    qc.compose( circ_0x, range(p), inplace=True )
                    qc.compose( circ_hk, range(p), inplace=True )
                    circs_state_s.append( qc )
            
            circs_measure_s = [circ_0, circ_h, circ_k ]    
            
            circ_state = []
            for j in range(n):
                list_qubits = range(j,n*p,n)
                if j == 0 :
                    for circ in circs_state_s:
                        qc0 = QuantumCircuit( n*p )
                        qc0.compose(circ.copy(), qubits=list_qubits, inplace=True)
                        circ_state.append(qc0)
                else:
                    circ_loop = circ_state.copy()
                    circ_state = []
                    for qc1 in circ_loop:
                        for circ in circs_state_s:
                            qc2 = qc1.compose(circ.copy(), qubits=list_qubits)
                            circ_state.append(qc2)
        
            circ_measure = []
            for j in range(n):
                list_qubits = range(j,n*p,n)
                if j == 0 :
                    for circ in circs_measure_s:
                        qc0 = QuantumCircuit( n*p )
                        qc0.compose(circ.copy(), qubits=list_qubits, inplace=True)
                        circ_measure.append(qc0)
                else:
                    circ_loop = circ_measure.copy()
                    circ_measure = []
                    for qc1 in circ_loop:
                        for circ in circs_measure_s:
                            qc2 = qc1.compose(circ.copy(), qubits=list_qubits)
                            circ_measure.append(qc2)     
            
            circs_mpt = []
            for i in range(6**n):
                for j in range(3**2):        
                    qc = QuantumCircuit( n*p, 2*n*p )
                    qc.compose( circ_state[i], qubits=range(n*p), inplace=True )
                    qc.measure( range(n*p), range(n*p) )
                    qc.compose( circ_measure[j], qubits=range(n*p), inplace=True )
                    qc.measure( range(n*p), range(n*p,2*n*p) )
                    circs_mpt.append( qc )
                
        self._circuits = circs_mpt
                   
        return circs_mpt
                  
        
    def fit( self, results, circuits=None, gate_set = None, resampling = 0, out = 0 ):         
                 
        if circuits is None :
            circuits = self._circuits
        elif self._p is None : 
            self._p  = int( circuits[0].num_qubits / self._n )
        
        if self._n == 1:
            self._counts = []
            for qc in circuits:
                self._counts.append( resampling_counts( results.get_counts(qc), resampling=resampling ) )
    
            if gate_set is None :
                self._gateset = False 
                self._states = np.array( [ [[1,0],[0,0]],
                              [[0,0],[0,1]],
                              [[1/2,1/2],[1/2,1/2]],
                              [[1/2,-1/2],[-1/2,1/2]],
                              [[1/2,-1j/2],[1j/2,1/2]],
                              [[1/2,1j/2],[-1j/2,1/2]],
                             ]).reshape( 6,4 ).T
    
                self._measurements = self._states / 3   
            else :
                self._gateset = True
                self._states, self._measurements = gate_set
                

            Υ_hat_all = []
            for m in range(self._p) :
                ran = [ m, self._p+m ] 
                probs = []
                for counts in self._counts:
                    probs_temp = dict2array( marginal_counts_dictionary( counts, ran ), 2 ) 
                    probs.append( probs_temp/np.sum(probs_temp)  )
                del probs_temp
                probs = np.array(probs).reshape([6,3,2,2]).transpose(0,1,3,2).reshape(6,6,2).transpose(1,0,2)/3
                if self._gateset is False :
                    Υ_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states, 
                                                                           self._measurements, 
                                                                           probs , Func = 0, 
                                                                           vectorized=True, out=out )
                elif self._gateset is True : 
                    Υ_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states[m], 
                                                                           self._measurements[m], 
                                                                           probs, Func = 0, 
                                                                           vectorized=True, out=out )
                Υ_hat_all.append( Υ_hat)
        else:  
            self._counts = []
            for qc in circuits:
                self._counts.append( resampling_counts( results.get_counts(qc), resampling=resampling ) )       
    
            if gate_set is None :
                self._gateset = False
                states_s = np.array( [ [[1,0],[0,0]],
                              [[0,0],[0,1]],
                              [[1/2,1/2],[1/2,1/2]],
                              [[1/2,-1/2],[-1/2,1/2]],
                              [[1/2,-1j/2],[1j/2,1/2]],
                              [[1/2,1j/2],[-1j/2,1/2]],
                             ])
    
                measures_s = states_s.reshape(3,2,2,2)
    
                states = []
                for s1 in states_s:
                    for s2 in states_s:
                        state_temp = np.kron( s1, s2 )
                        states.append( state_temp.flatten() )
                self._states = np.array(states).T
    
                measures = []
                measures_temp = np.kron( measures_s[0,0], measures_s[0,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,0], measures_s[0,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,1], measures_s[0,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,1], measures_s[0,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,0], measures_s[1,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,0], measures_s[1,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,1], measures_s[1,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,1], measures_s[1,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,0], measures_s[2,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,0], measures_s[2,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,1], measures_s[2,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[0,1], measures_s[2,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,0], measures_s[0,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,0], measures_s[0,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,1], measures_s[0,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,1], measures_s[0,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,0], measures_s[1,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,0], measures_s[1,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,1], measures_s[1,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,1], measures_s[1,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,0], measures_s[2,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,0], measures_s[2,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,1], measures_s[2,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[1,1], measures_s[2,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,0], measures_s[0,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,0], measures_s[0,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,1], measures_s[0,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,1], measures_s[0,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,0], measures_s[1,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,0], measures_s[1,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,1], measures_s[1,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,1], measures_s[1,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,0], measures_s[2,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,0], measures_s[2,1] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,1], measures_s[2,0] )
                measures.append( measures_temp.flatten() )   
                measures_temp = np.kron( measures_s[2,1], measures_s[2,1] )
                measures.append( measures_temp.flatten() )   
                self._measures = np.array(measures).T/9    
                
                
            else :
                self._gateset = True
                states_s, measures_s = gate_set 
                measures_s = np.array(measures_s).reshape(self._n*self._p,4,3,2).transpose(0,2,3,1)
                
                self._states = []
                self._measures = []
                
                for m in range(self._p):
                    
                    states = []
                    for s1 in range(6):
                        for s2 in range(6):
                            state_temp = qt.Outer2Kron( np.kron( states_s[m*self._n][:,s1], 
                                                                states_s[m*self._n+1][:,s2] ), [2,2] )
                            states.append( state_temp.flatten() )
                    self._states.append( np.array(states).T )
    
    
                    measures = []
                    for r1 in range(3):
                        for r2 in range(3):
                            for s1 in range(2):
                                for s2 in range(2):
                                    measures_temp = qt.Outer2Kron( np.kron( measures_s[m*self._n,r1,s1], 
                                                                           measures_s[m*self._n+1,r2,s2] ), [2,2] )
                                    measures.append( measures_temp.flatten() )   
                    self._measures.append( np.array(measures).T ) 
            
            Υ_hat_all = []
            for m in range(self._p) :
                ran = [ self._p, self._p+m]
                probs = []
                for counts in self._counts:
                    
                    probs_temp = dict2array( marginal_counts_dictionary( counts, ran ), 4 ) 
                    probs.append( probs_temp/np.sum(probs_temp)  )
                del probs_temp
                probs_loop = np.array(probs).reshape(36,9,4,4).transpose(0,1,3,2).reshape(6**self._n,
                                                                                          6**self._n,
                                                                                          2**self._n).transpose(1,0,2)/3**2
                if m == 0 :
                    print( marginal_counts_dictionary( counts, ran ) )
                    print( probs_loop )

                if self._gateset is False :
                    Υ_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states, 
                                                                           self._measures, 
                                                                           probs_loop, Func = 0, 
                                                                           vectorized=True, out=out )
                elif self._gateset is True :    
                    Υ_hat = qt.MaximumLikelihoodCompleteDetectorTomography( self._states[m], 
                                                                           self._measures[m], 
                                                                           probs_loop, Func = 0, 
                                                                           vectorized=True, out=out )
                Υ_hat_all.append( Υ_hat ) 
        
        if len(Υ_hat_all) == 1 :
            return Υ_hat_all[0]
        else:
            return Υ_hat_all
               


############# Device Tomography ################

class device_process_measurement_tomography :
    
    def __init__( self, backend, max_qobj=900 ) :
        
        self._backend    = backend
        self._num_qubits = len( backend.properties().qubits )
        self._max_qobj   = max_qobj
        
        coupling_map = get_backend_conectivity( self._backend )
    
        G = nx.Graph()
        G.add_node( range(self._num_qubits) )
        G.add_edges_from(coupling_map)
        G = nx.generators.line.line_graph(G)
        G_coloring = nx.coloring.greedy_color(G)
        degree = max( G_coloring.values() ) + 1
        parall_qubits = degree*[None]
        for x in G_coloring:
            if parall_qubits[G_coloring[x]] is None:
                parall_qubits[G_coloring[x]] = []
            parall_qubits[G_coloring[x]].append(x)
    
            
        circs_all = [ tomographic_gate_set_tomography( self._num_qubits ).circuits(), 
                     measurement_process_tomography( 1, self._num_qubits ).circuits() ]
    
        for pairs in parall_qubits :
    
            p = len(pairs)
            qubits = pairs
            qubits = [item for t in qubits for item in t]
            circ_double = measurement_process_tomography( 2, p ).circuits()
            circs = []
            for circ_loop in circ_double:
                circ = QuantumCircuit( self._num_qubits, 4*p )
                circ.compose(circ_loop, qubits=qubits, inplace=True)
                circs.append( circ )
            circs_all.append( circs )
            
        circs_pkg = []
        circ_temp = []
        pkg_idx = []
    
        idx = 0
        for circs in circs_all:
            if len(circ_temp) + len(circs) <= self._max_qobj:
                circ_temp += circs.copy()
                pkg_idx.append( idx ) 
            else :
                circs_pkg.append( circ_temp )
                circ_temp = circs.copy()
                idx += 1
        circs_pkg.append( circ_temp )
        pkg_idx.append( idx ) 
        
        self._circs_all  = circs_all
        self._circs_pkg     = circs_pkg
        self._pkg_idx       = pkg_idx
        self._parall_qubits = parall_qubits
        

    def circuits( self ):
        """
        Circuits to perform the process measurement tomography of each pair of connected qubits on a device
        
        In:
            backend
        out:
            circs_pkg : efficient storage of the circuits for execution.
        
        """
        
        return self._circs_pkg 
    
    
    def fit( self, results, out=1, resampling=0, paralell=True ):
        
        gateset = tomographic_gate_set_tomography( self._num_qubits ).fit( results[self._pkg_idx[0]] , 
                                                         self._circs_all[0], 
                                                         resampling = resampling )
            
        states_gst= []
        measures_gst = []
        for m in range(self._num_qubits):
            rho = gateset[0][m]
            Pi  = gateset[1][m]
            Y   = gateset[2][m]
            states_gst_temp   = []
            measures_gst_temp = []
            for v in [ np.eye(4), Y[2], Y[3] ]:
                for u in [ np.eye(4), Y[1] ]:
                    states_gst_temp.append( v@u@rho )
                measures_gst_temp.append( v.T.conj()@Pi )    
    
            states_gst.append( np.array(states_gst_temp).T )
            measures_gst.append( np.array(measures_gst_temp).transpose(1,0,2).reshape(4,-1)/3 )
    
        states_gst   = np.array( states_gst )
        measures_gst = np.array( measures_gst )
    
        choi_single = measurement_process_tomography(1,self._num_qubits).fit( results[self._pkg_idx[1]], 
                                                           self._circs_all[1], 
                                                           resampling=resampling,
                                                           out = out)
    
        if paralell is False:
            choi_double = []
            for k in range(2,len(self._circs_all)) :
                choi_double.append( measurement_process_tomography(2).fit( results[self._pkg_idx[k]], 
                                                                         self._circs_all[k], 
                                                                         resampling=resampling, 
                                                                         out = out  ) )
        elif paralell is True:
            fun_par = lambda k : measurement_process_tomography(2).fit( results[self._pkg_idx[k]], 
                                                                      self._circs_all[k], 
                                                                      resampling=resampling,
                                                                      out = out )
            choi_double = Parallel(n_jobs=-1)( delayed( fun_par )(k) for k in range(2,len(self._circs_all)) ) 
        
        return choi_single, choi_double, gateset 
                 

            



############# Noise model ################

def decoherence_noise( T1=5e3, T2=200e3 ):

    # T1 and T2 values for qubits 0-3
    T1s = np.random.normal( T1, np.sqrt(T1), 7) # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal( T2, np.sqrt(T2), 7)  # Sampled from normal distribution mean 50 microsec
    
    # Truncate random T1s <= 0
    T1s[T1s<0]=0
    
    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(7)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                      for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                  for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                 thermal_relaxation_error(t1b, t2b, time_cx))
                  for t1a, t2a in zip(T1s, T2s)]
                   for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()
    noise_thermal.add_quantum_error(errors_reset[0], "reset", [0])
    noise_thermal.add_quantum_error(errors_measure[0], "measure", [0])
    noise_thermal.add_quantum_error(errors_u1[0], "u1", [0])
    noise_thermal.add_quantum_error(errors_u2[0], "u2", [0])
    noise_thermal.add_quantum_error(errors_u3[0], "u3", [0])
    noise_thermal.add_quantum_error(errors_cx[0][0], "cx", [0, 0])
    noise_thermal.add_quantum_error(errors_cx[0][1], "cx", [0, 1])
    noise_thermal.add_quantum_error(errors_cx[0][2], "cx", [0, 2])
    noise_thermal.add_quantum_error(errors_cx[0][3], "cx", [0, 3])
    noise_thermal.add_quantum_error(errors_reset[1], "reset", [1])
    noise_thermal.add_quantum_error(errors_measure[1], "measure", [1])
    noise_thermal.add_quantum_error(errors_u1[1], "u1", [1])
    noise_thermal.add_quantum_error(errors_u2[1], "u2", [1])
    noise_thermal.add_quantum_error(errors_u3[1], "u3", [1])
    noise_thermal.add_quantum_error(errors_cx[1][0], "cx", [1, 0])
    noise_thermal.add_quantum_error(errors_cx[1][1], "cx", [1, 1])
    noise_thermal.add_quantum_error(errors_cx[1][2], "cx", [1, 2])
    noise_thermal.add_quantum_error(errors_cx[1][3], "cx", [1, 3])
    noise_thermal.add_quantum_error(errors_reset[2], "reset", [2])
    noise_thermal.add_quantum_error(errors_measure[2], "measure", [2])
    noise_thermal.add_quantum_error(errors_u1[2], "u1", [2])
    noise_thermal.add_quantum_error(errors_u2[2], "u2", [2])
    noise_thermal.add_quantum_error(errors_u3[2], "u3", [2])
    noise_thermal.add_quantum_error(errors_cx[2][0], "cx", [2, 0])
    noise_thermal.add_quantum_error(errors_cx[2][1], "cx", [2, 1])
    noise_thermal.add_quantum_error(errors_cx[2][2], "cx", [2, 2])
    noise_thermal.add_quantum_error(errors_cx[2][3], "cx", [2, 3])
    noise_thermal.add_quantum_error(errors_reset[3], "reset", [3])
    noise_thermal.add_quantum_error(errors_measure[3], "measure", [3])
    noise_thermal.add_quantum_error(errors_u1[3], "u1", [3])
    noise_thermal.add_quantum_error(errors_u2[3], "u2", [3])
    noise_thermal.add_quantum_error(errors_u3[3], "u3", [3])
    noise_thermal.add_quantum_error(errors_cx[3][0], "cx", [3, 0])
    noise_thermal.add_quantum_error(errors_cx[3][1], "cx", [3, 1])
    noise_thermal.add_quantum_error(errors_cx[3][2], "cx", [3, 2])
    noise_thermal.add_quantum_error(errors_cx[3][3], "cx", [3, 3])
    noise_thermal.add_quantum_error(errors_reset[4], "reset", [4])
    noise_thermal.add_quantum_error(errors_measure[4], "measure", [4])
    noise_thermal.add_quantum_error(errors_u1[4], "u1", [4])
    noise_thermal.add_quantum_error(errors_u2[4], "u2", [4])
    noise_thermal.add_quantum_error(errors_u3[4], "u3", [4])
    noise_thermal.add_quantum_error(errors_cx[4][0], "cx", [4, 0])
    noise_thermal.add_quantum_error(errors_cx[4][1], "cx", [4, 1])
    noise_thermal.add_quantum_error(errors_cx[4][2], "cx", [4, 2])
    noise_thermal.add_quantum_error(errors_cx[4][3], "cx", [4, 3])
    noise_thermal.add_quantum_error(errors_reset[5], "reset", [5])
    noise_thermal.add_quantum_error(errors_measure[5], "measure", [5])
    noise_thermal.add_quantum_error(errors_u1[5], "u1", [5])
    noise_thermal.add_quantum_error(errors_u2[5], "u2", [5])
    noise_thermal.add_quantum_error(errors_u3[5], "u3", [5])
    noise_thermal.add_quantum_error(errors_cx[5][0], "cx", [5, 0])
    noise_thermal.add_quantum_error(errors_cx[5][1], "cx", [5, 1])
    noise_thermal.add_quantum_error(errors_cx[5][2], "cx", [5, 2])
    noise_thermal.add_quantum_error(errors_cx[5][3], "cx", [5, 3])
    noise_thermal.add_quantum_error(errors_reset[6], "reset", [6])
    noise_thermal.add_quantum_error(errors_measure[6], "measure", [6])
    noise_thermal.add_quantum_error(errors_u1[6], "u1", [6])
    noise_thermal.add_quantum_error(errors_u2[6], "u2", [6])
    noise_thermal.add_quantum_error(errors_u3[6], "u3", [6])
    noise_thermal.add_quantum_error(errors_cx[6][0], "cx", [6, 0])
    noise_thermal.add_quantum_error(errors_cx[6][1], "cx", [6, 1])
    noise_thermal.add_quantum_error(errors_cx[6][2], "cx", [6, 2])
    noise_thermal.add_quantum_error(errors_cx[6][3], "cx", [6, 3])
    return noise_thermal    





############ Quantities ###################

def readout_fidelity( Pi ):
    d, N = Pi.shape
    d = int(np.sqrt(d))
    f = 0.
    for n in range(N):
        f += Pi[:,n].reshape(d,d)[n,n]/N
    return np.real( f )  

def qnd_fidelity( choi ):
    N = len(choi)
    d = int(np.sqrt(choi[0].shape[0]))
    f = 0
    for n in range(N):
        f += choi[n][(1+d)*n,(1+d)*n]/N
    return np.real( f )     
    
def destructiveness( chois ):
    choi = np.sum( chois, axis=0 )
    d = int(np.sqrt(choi.shape[0]))
    if d == 2:
        O = np.array([1,0,0,-1])/np.sqrt(2)
        D = np.linalg.norm( O - choi.T.conj()@O )/np.sqrt(8)
    else:
        P = np.eye(d)
        Bs = np.zeros((d**2,d),dtype=complex)
        for k in range(d):
            pp = np.kron( P[:,k], P[:,k] )
            Bs[:,k] = pp - choi.T.conj()@pp 
        B = Bs.T.conj()@Bs
        vals, vecs = np.linalg.eigh(B)
        D = 0.5 * np.sqrt( np.max(vals)/2 )
    return D    
    
def Quantities( Pi, choi ):
    
    f = readout_fidelity( Pi )
    q = qnd_fidelity( choi )
    d = 1 - destructiveness( choi )
    
    return f, q, d
            
def Kron_Choi( Choi_1, Choi_2 ):
    Y0 = [] 
    for i in range( len(Choi_1) ):
        for j in range(len(Choi_2)):
            Y_loop = np.kron( Choi_1[i],  Choi_2[j]) 
            Y_loop =  Y_loop.reshape(8*[2]).transpose(0,2,1,3,4,6,5,7).reshape(16,16) 
            Y0.append( Y_loop )
    return Y0

def Cross_QNDness( Choi_single_1, Choi_single_2, Choi_double  ):
    Y0 = [ qt.Process2Choi( A )/2 for A in Kron_Choi( Choi_single_1, Choi_single_2 )]
    Y1 = [ qt.Process2Choi( A )/2 for A in Choi_double]
    f = 0
    f += qt.Fidelity( Y0[0], Y1[0] )/2
    f += qt.Fidelity( Y0[1], Y1[1] )/2
    f += qt.Fidelity( Y0[2], Y1[2] )/2
    f += qt.Fidelity( Y0[3], Y1[3] )/2
    return f

def Cross_Fidelity( Pi_single_1, Pi_single_2, Pi_double  ):
    Pi0 = [ np.kron(A,B)/2 for A in Pi_single_1.reshape(2,2,2).transpose(1,2,0) for B in Pi_single_2.reshape(2,2,2).transpose(1,2,0) ]
    Pi1 = Pi_double.reshape(4,4,4).transpose(1,2,0)/2
    f = 0
    f += qt.Fidelity( Pi0[0], Pi1[0] )/2
    f += qt.Fidelity( Pi0[1], Pi1[1] )/2
    f += qt.Fidelity( Pi0[2], Pi1[2] )/2
    f += qt.Fidelity( Pi0[3], Pi1[3] )/2
    return f

def Cross_Quantities( Pi1, Choi1, Pi2, Choi2, Pi12, Choi12 ):
    
    f = Cross_Fidelity( Pi1, Pi2, Pi12 )
    q = Cross_QNDness( Choi1, Choi2, Choi12 )
    
    return f, q

####################### Plots ############################

def sph2cart(r, theta, phi):
    '''spherical to Cartesian transformation.'''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def sphview(ax):
    '''returns the camera position for 3D axes in spherical coordinates'''
    r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90-ax.elev, ax.azim))
    return r, theta, phi

def getDistances(view, xpos, ypos, dz):
    distances  = []
    for i in range(len(xpos)):
        distance = (xpos[i] - view[0])**2 + (ypos[i] - view[1])**2 + (dz[i] - view[2])**2
        distances.append(np.sqrt(distance))
    return distances

def Bar3D( A , ax = None, xpos=None, ypos=None, zpos=None, dx=None, dy=None, M = 0, **args ):
    
    d = A.shape[0]
    camera = np.array([13.856, -24. ,0])
    
    if xpos is None :
        xpos = np.arange(d) 
    if ypos is None :
        ypos = np.arange(d)
    xpos, ypos = np.meshgrid( xpos, ypos )
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    
    if zpos is None :
        zpos = np.zeros_like(xpos)
    else :
        zpos = zpos.flatten()
    
    if dx is None :
        dx = 0.5 * np.ones_like(xpos)
    else :
        dx = dx * np.ones_like(ypos)
        
    if dy is None :
        dy = 0.5 * np.ones_like(ypos)
    else :
        dy = dy * np.ones_like(ypos)
    
    dz = A.flatten()
    z_order = getDistances(camera, xpos, ypos, zpos)
    
    if ax == None :
        fig = plt.figure()   
        ax  = fig.add_subplot( 1,1,1, projection='3d')  
    maxx    = np.max(z_order) + M
    
#     plt.rc('font', size=15) 
    for i in range(xpos.shape[0]):
        pl = ax.bar3d(xpos[i], ypos[i], zpos[i], 
                      dx[i], dy[i], dz[i], 
                      zsort='max', **args )
        pl._sort_zpos = maxx - z_order[i]
#        ax.set_xticks( [0.25,1.25,2.25,3.25] )
#        ax.set_xticklabels((r'$|gg\rangle$',r'$|ge\rangle$',
#                                r'$|eg\rangle$',r'$|ee\rangle$'))
#        ax.set_yticks( [0.25,1.25,2.25,3.25] )
#        ax.set_yticklabels((r'$\langle gg|$',r'$\langle ge|$',
#                                r'$\langle eg|$',r'$\langle ee|$'))
#         ax.set_title( label, loc='left', fontsize=20, x = 0.1, y=.85)
    ax.set_zlim([0,1])
    return ax            


def Abs_Bars3D(Y):
    fig = plt.figure(figsize=(len(Y)*4,5)) 
    for y in range(len(Y)):
        ax  = fig.add_subplot( 1, len(Y), y+1,  projection='3d')
        Bar3D( np.abs( Y[y] ).T, ax=ax )   
    return fig



            
            
            
            
            
            
            
            
            
            
