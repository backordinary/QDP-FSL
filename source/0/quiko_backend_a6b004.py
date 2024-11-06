# https://github.com/scottoshiro2/QuiKo-v2.0/blob/eb9a13d6bd486d4bc052086ba6b353056f72d1b6/QuiKo_Backend.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:09:42 2021

@author: scottoshiro
## QuiKo Circuit Generation Function

## Need an automatic database 
## Phase KickBack Sequencing Encoding Method
## - This inlcudes 
"""

from librosa import *

import math
import numpy as np
from math import pi

import matplotlib.pyplot as plt

from qiskit import IBMQ
from qiskit import QuantumCircuit, Aer, assemble, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit 

## Will need to import the analysis library from here this will connect to the results from
## running the Quantum Circuit

print('Testing Version...')
print('Loading IBM Account...')
#IBMQ.load_account()     
IBMQ.save_account('09fc89cbe6abe2504689109464ad5f8c2e060e429246b5f957306214948b776046521a6b7191834c17c98db00c20801655fbb368ce16c166992af6b1f4bc0f11')
IBMQ.enable_account(token='09fc89cbe6abe2504689109464ad5f8c2e060e429246b5f957306214948b776046521a6b7191834c17c98db00c20801655fbb368ce16c166992af6b1f4bc0f11',hub='ibm-q-academic', group='stanford', project='q-comp-music-gen')
#IBMQ.load_account() 
#print(IBMQ.enable_account(token='09fc89cbe6abe2504689109464ad5f8c2e060e429246b5f957306214948b776046521a6b7191834c17c98db00c20801655fbb368ce16c166992af6b1f4bc0f11',hub='ibm-q-academic', group='stanford', project='q-comp-music-gen'))
##----------------------------------------------------------------------------

class Encoding_matrix:
    
    def __init__( self, feature_matrix, subdivision, num_bands, qubit_budget ):
        """
        This class is for the extraction and parsing of musical/signal features
        The parsing and organization is for the easy encoding onto quantum circuits
        """
        self.subdivision = subdivision
        self.num_bands = num_bands
        self.qubit_budget = qubit_budget
        self.feature_matrix = feature_matrix
        self.pkbse_matrix = []
        
        self.num_bands = num_bands
        self.subdivision = subdivision
        
    def APencoding_matrix( self ):
        """
        This should be of the size 3X3, no subdivision data
        """
        return self.feature_matrix
        
    def static_encoding_matrix( self ):
        """
        This function is to set up the encoding scheme that follows the FRQA method
        Note: This is to be compared to pksbe and the difference in rhythm...
              Phase kickback has less of a significant effect to be leveraged
        """
        return self.feature_matrix ## This should have more dimensions to the matrix for each subdiv
    
    def pkbse_encoding_matrix( self ):
        """
        Description:
        This parses the feature set into a matrix easily enconded using the pkbse method
        Note: Might be able to control the overall 'response window'
        
        Note: We are thinking of the subdivisions interms on if we were improvising. We make musical decisions based on
              what has already happened...influence. The more the subdivisions progress we have to take into account the influences of
              the previous subdivisions.
        """
        ##...Parsing matrix before next line...
        ## This need to be generalized for any size feature and gate set
        aggr_array0 = []; aggr_array02 = []; aggr_array03 = []
        aggr_array1 = []; aggr_array12 = []; aggr_array13 = []
        aggr_frame0 = 0;  aggr_frame02 = 0;  aggr_frame03 = 0
        aggr_frame1 = 0;  aggr_frame12 = 0;  aggr_frame13 = 0
        
        for band in range( self.num_bands ):
            #print( 'band: ', band, self.feature_matrix[band] )
            #print( '  ' )
            feature = self.feature_matrix[band]
            dataset_size = int( len( feature ) / 2 )
            
            for subdiv in range( len( feature ) ): ## This needs to work for any dim of feature set
                if subdiv < dataset_size:
                    #print( 'check point...', subdiv )
                    aggr_frame0  += feature[subdiv][0]
                    aggr_frame02 += feature[subdiv][1]
                    aggr_frame03 += feature[subdiv][2]
                    
                    if subdiv != (dataset_size - 1):
                        aggr_array0.append(-1.0*round(aggr_frame0,3))
                        aggr_array02.append(-1.0*round(aggr_frame02,3))
                        aggr_array03.append(-1.0*round(aggr_frame03,3))
                    else:
                        aggr_array0.append(round(aggr_frame0,3))
                        aggr_array02.append(round(aggr_frame02,3))
                        aggr_array03.append(round(aggr_frame03,3))
                else:
                    #print( 'check point...on the 1 side', subdiv )
                    aggr_frame1  += feature[subdiv][0]
                    aggr_frame12 += feature[subdiv][1]
                    aggr_frame13 += feature[subdiv][2]
                    
                    if subdiv != (2*dataset_size - 1):
                        aggr_array1.append(-1.0*round(aggr_frame1,3))
                        aggr_array12.append(-1.0*round(aggr_frame12,3))
                        aggr_array13.append(-1.0*round(aggr_frame13,3))
                    else:
                        aggr_array1.append(round(aggr_frame1,3))
                        aggr_array12.append(round(aggr_frame12,3))
                        aggr_array13.append(round(aggr_frame13,3))
            
            pkb_band = [ [list(np.flip(aggr_array0)),  list(np.flip( aggr_array1))],
                         [list(np.flip(aggr_array02)), list(np.flip( aggr_array12))], 
                         [list(np.flip(aggr_array03)), list(np.flip( aggr_array13))] ] 
            
            self.pkbse_matrix.append( pkb_band )
            
            aggr_array0 = []; aggr_array02 = []; aggr_array03 = []
            aggr_array1 = []; aggr_array12 = []; aggr_array13 = []
            aggr_frame0 = 0;  aggr_frame02 = 0;  aggr_frame03 = 0
            aggr_frame1 = 0;  aggr_frame12 = 0;  aggr_frame13 = 0
                      
        return self.pkbse_matrix
##----------------------------------------------------------------------------

class QuikoCircuit: ## This needs a new name for better description
    
    def __init__( self, num_bands, spine_qubits, cbits, encoding_matrix, Encoding_method, big_circ=False ):
        self.big_circ = big_circ
        self.num_bands = num_bands
        self.Encoding_method = Encoding_method
        self.spine_qubits = spine_qubits ## This specifies the number of qubits on the
        self.subdivisions = 2**self.spine_qubits
        
        self.encoding = encoding_matrix#; print(self.encoding)
        self.timbre_qubits = num_bands 

        if self.Encoding_method == 0:
            #if big_circ == False:
            self.qc = QuantumCircuit( self.timbre_qubits )
        else:
            if self.big_circ == False:
                self.qc = QuantumCircuit( self.timbre_qubits + self.spine_qubits, cbits )
            else:
                self.qc = QuantumCircuit( self.timbre_qubits + self.spine_qubits )
            
    def internal_pulse( self, statevector=[None,None] , static_timbre=False ):
        """
        This function is to initialize interal pulse of the system
        The default is going to be in super position, The static is to put the static code on the timbre register
        Note: If we put the timbre register in superposition does it have a higher expressibility?
              Should I use this metric?
              
        This should also be combines with pkbse so there is no confusion with setting it up with the rest of the circuit.
        statevector should an array of statevectors for the two registers[0] timbre, [1] spine
        """
        
        ## This is for the timbre qubits
        if statevector[0] == None:
            if static_timbre == True:
                self.qc.x( range(self.timbre_qubits) )
            else:
                self.qc.h( range(self.timbre_qubits) )
        else:
            self.qc.initialize( statevector[0], range(self.timbre_qubits) ) ## This is the last place I left off...
            
        ## This is for the spinal cord qubits
        if statevector[1] == None:
            self.qc.h( range(self.timbre_qubits, self.timbre_qubits + self.spine_qubits) )
        else:
            self.qc.initialize( statevector[1], range(self.timbre_qubits, self.timbre_qubits + self.spine_qubits) )

        return self.qc
    
    def amplitude_phase_encoding( self ):
        """
        This is for the database audio samples, not spinal cord regsiter will be present
        This should be triggered if self.half_circ is False (change the name)
        
        Note: for initial analysis we do not want to add an internal pulse to the 
              audio samples in the data base => no h gates on the timbre qubits
        """
        for band, params in enumerate( self.encoding ):
            ## Internal pulse can go here as well
            self.qc.u3( params[0], params[1], params[2], band )
        
        return self.qc
    
    def static_encoding( self ):
        """
        This is for the multi-control u3 gates, this will be compared to pkbse_encoding
        Take straight from the jupyter notebooks
        
        might need to do this one subdiv at a time when on real device...
        """
        #print( self.encoding )
        for subdiv in range( 2**self.spine_qubits ):
                subdiv_str = format(subdiv, "0{}b".format(self.spine_qubits))
                
                for index, bit in enumerate( subdiv_str ):
                    
                    if bit != '1':
                        #print(subdiv_str, self.timbre_qubits + index)
                        self.qc.x( self.timbre_qubits + index )
                        
                for band, seq in enumerate( self.encoding ):    
                    for subdiv, param in enumerate( seq ):
                        qc_u = QuantumCircuit( 1 )
                        qc_u.u3( param[0], param[1], param[2], 0 )
                        qc_u = qc_u.control( self.spine_qubits )

                    qubit_ctrl = [q + self.num_bands for q in range(self.spine_qubits)]
                    qubit_ctrl.append(band)  
                    #print(qubit_ctrl)  
                    self.qc = self.qc.compose( qc_u, qubit_ctrl )
                
                for index, bit in enumerate( subdiv_str ):
                    if bit != '1':
                        self.qc.x( self.timbre_qubits + index )           
       
            
    def pkbse_encoding( self ):
        """
        features is a multi-dim matrix 3 X 8 with each element size [3]
        gate_set = list of gates that user can specify for encoding method
        
        So far this is made for a three qubit spinal cord system, I forgot to negate the following angles
        """
        total_qubits = self.timbre_qubits + self.spine_qubits
        if 0 == 0:
            self.qc.x( range(total_qubits - self.spine_qubits, total_qubits) )
        
        for band, band_elements in enumerate( self.encoding ):
            for subdivi in range( int((2**self.spine_qubits) / 2) ):
                gate_params = []
                for param_num, param in enumerate( band_elements ):
                    gate_params.append( self.encoding[band][param_num][0][subdivi] )
                    
                self.qc.cu3( gate_params[0], gate_params[1], gate_params[2], 
                             total_qubits - (self.spine_qubits - band), band ) 
           
        if 0 == 0:
            self.qc.x( range(total_qubits - self.spine_qubits, total_qubits) )
        if 1 == 0:
            self.qc.x( range(total_qubits - self.spine_qubits, total_qubits) )
        
        for band, band_elements in enumerate( self.encoding ):
            for subdivi in range( int((2**self.spine_qubits) / 2) ):
                gate_params = []
                for param_num, param in enumerate( band_elements ):
                    gate_params.append( self.encoding[band][param_num][1][subdivi] )
                    
                self.qc.cu3( gate_params[0], gate_params[1], gate_params[2], 
                             total_qubits - (self.spine_qubits - band), band ) 
           
        if 1 == 0:
            self.qc.x( range(total_qubits - self.spine_qubits, total_qubits) )
    def Quantum_Circuit( self, Encoding_method, meas_map=[[],[]] ):
        """
        This is the built-in quantum circuit generator. Users can rearrange the
        components that suits thier designs
        """
        total_qubits = self.timbre_qubits + self.spine_qubits
        
        
        if Encoding_method == 0:
            ## This is perserved for the audio samples in the database
            self.amplitude_phase_encoding()
            
        elif Encoding_method == 1:
            self.internal_pulse( static_timbre=False )
            self.pkbse_encoding()
            #print('dagger range: ', range(self.timbre_qubits, total_qubits))
            qft_inv = QuantumCircuit( self.spine_qubits )
            self.qft_dagger( qft_inv, self.spine_qubits )
            self.qc = self.qc.compose( qft_inv, range(self.timbre_qubits, total_qubits ) )
            
        else:
            self.internal_pulse( static_timbre=False )
            self.static_encoding()
        
        """ 
        Do keep in mind here that we should make a multi-dimensional description
        of the quantum states and measure it in different basis (x,y, and z)
        
        """
        if meas_map != []:
            if self.big_circ == False:
                self.qc.measure(meas_map[0],meas_map[1])
        else:
            if self.big_circ == False:
                self.qc.measure_all()
    

        return self.qc
    
    def basis_measure( self, basis_meas=['x',None,None] ):
        """
        This function is to set up the different measurement basis for analysis
        """
        if basis_meas[0] == 'x':
            self.qc.measure_all()    
        elif basis_meas[1] == 'y':
            self.qc.measure_all()
        else:
            self.qc.measure_all()
        
        pass
    
    def qft_dagger(self, qcirc, n):
        """n-qubit QFTdagger the first n qubits in circ
        This code of tthe qft_dagger function is from the qiskit online textbook
        
        """
        # Don't forget the Swaps!
        for qubit in range(n//2):
            #print(qubit)
            #qubit += 3
            #print(qubit)
            qcirc.swap(qubit, n-qubit-1)
            
        for j in range(n):
            #j += 3
            for m in range(j):
                #m += 3
                #print(m, j)
                qcirc.cp(-math.pi/float(2**(j-m)), m, j)
            qcirc.h(j)
            
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
def setup_qc( features, encoding_method, big_circ=True  ):
    num_bands = 3
    if encoding_method == 1:
        pkbse = 1
        Encoding_method = 1
        
        spine_qubits = num_bands; cbits = 2*num_bands
        
        #meas_map = [meas_m,meas_m]
        parameters = Encoding_matrix( features, 2**num_bands, num_bands, 2**num_bands)
        pkbse_matrix = parameters.pkbse_encoding_matrix()
        #static_matrix = parameters.static_encoding_matrix()
        
        qcirc = QuikoCircuit( num_bands, spine_qubits, cbits, pkbse_matrix, Encoding_method, big_circ=True )
        qc = qcirc.Quantum_Circuit( Encoding_method,meas_map=[]) #meas_map )

    elif encoding_method == 2:
        pkbse = 1
        Encoding_method = 2
        
        spine_qubits = num_bands; cbits = 2*num_bands
        
        meas_map = [meas_m,meas_m]
        parameters = Encoding_matrix( features, 2**num_bands, num_bands, 2**num_bands)
        #pkbse_matrix = parameters.pkbse_encoding_matrix()
        static_matrix = parameters.static_encoding_matrix()
        
        qcirc = QuikoCircuit( num_bands, spine_qubits, cbits, static_matrix, Encoding_method, big_circ=True )
        qc = qcirc.Quantum_Circuit( Encoding_method, meas_map=[])#meas_map=meas_map )
        
    else:
        print( 'invalid encoding method!' )
        qc = 'invalid method...'
        
    return qc
        
def Full_Quantum_Circuit( qc, database_bundle, encoding_method ):
    """

    Hard coded for IBMQ Montreal (27 qubits)
    
    """
    
    #qc = setup_qc( features, encoding_method )
    
    tracks = []
    for i in database_bundle:
        tracks.append(i)
    db_size = len(tracks)
    
    ancilla = 3
    qubit_num = 27
    
    db_start = 6
    
    qreg = QuantumRegister( 27 )
    creg_11 = ClassicalRegister( 138 ) #3 * len(tracks) + 3 ) #compare bit 1
     # bit 3
    
    creg_2 = 3 #subdivision
    qc2 = QuantumCircuit( qreg, creg_11 )
    
    new_qc = qc2.compose(qc)
    #new_qc = qc2.compose(database_bundle[0], [6,7,8])
    count = -1
    prev_samp = -1
    cbit_shift = 0
    rec = 0 
    for j, title in enumerate( tracks ):
        count += 1
        if count == 11:
            prev_samp = j + 1
            next_samp = prev_samp + 11
            row_entries = tracks[prev_samp:next_samp]
            
            for index, row in enumerate( row_entries ):
                new_qc = new_qc.compose(database_bundle[row], [db_start,db_start+1,db_start+2])
    
                new_qc.cx(0,24)
                new_qc.cx(db_start,24)
                
                new_qc.cx(1,25)
                new_qc.cx(db_start+1,25)
                
                new_qc.cx(2,25)
                new_qc.cx(db_start+2,26)
                
                new_qc.measure([24,25,26],[cbit_shift,cbit_shift+1,cbit_shift+2])
                cbit_shift += 3
                
                if index < len( row_entries ) - 1:
                    new_qc.reset([db_start,db_start+1,db_start+2])
                    
            db_start += 3
            count = -1
            
    #print( cbit_shift, rec )
    cend = 138 #3*len(tracks) + 3 #This will be hard coded for now!
    new_qc.measure( [3,4,5],[135,136,137] )
    
    return new_qc
        
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
if __name__ == "__main__":
    """
    Testing script with random input matrices
    """
    
    feature = [ [1,2,3],[4,5,6],[7,8,9],[10,11,12],
                [13,14,15],[16,17,18],[19,20,21],[22,23,24] ]
    
    data_base_feature = [ [1,2,3,4],[5,6,7,8],[9,10,11,12] ]
    
    
    feature_matrix = [ feature, feature, feature ]
    
    x = Encoding_matrix( feature_matrix, 8, 3, 3 )
    etype = 1 
    y = x.pkbse_encoding_matrix()
    print( etype, y )
    #y = x.pkbse_encoding_matrix()
    
    qkc = QuikoCircuit( 3, 3, 6, y, etype )
    quiKoc = qkc.Quantum_Circuit( etype )
    quiKoc.draw( output='mpl' )
    
    
    """
    xd = Encoding_matrix( data_base_feature, 8, 3, 3 )
    etype, yd = xd.APencoding_matrix()
    
    qkc = QuikoCircuit( 3, 3, yd, Encoding_method=etype )
    quiKoc = qkc.Quantum_Circuit( Encoding_method=etype )
    
    distr = execute_quantumcircuit( quiKoc, dev='aer_simulator' )
    quiKoc.draw( output='mpl' )
    """
    plt.show()
    
    print(y)
    