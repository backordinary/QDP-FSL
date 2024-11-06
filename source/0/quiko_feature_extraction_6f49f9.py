# https://github.com/scottoshiro2/QuiKo-v2.0/blob/eb9a13d6bd486d4bc052086ba6b353056f72d1b6/Quiko_Feature_extraction.py

import librosa
import numpy as np
from numpy import inf
from math import pi

import scipy.stats
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from scipy.signal import find_peaks
from scipy import signal
from scipy.signal import butter, lfilter, freqz

from QuiKo_Backend import *

from qiskit import IBMQ
from qiskit import QuantumCircuit, Aer, assemble, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex

## Feature Extraction Functions ----------------------------------------------------------------
def ifnan( harm_est, spec_cent, perc_est ):
    if np.isnan(harm_est) == True:
        harm_est = 2*np.pi
    if np.isnan(spec_cent) == True:
        spec_cent = 2*np.pi
    if np.isnan(perc_est) == True:
        perc_est = 2*np.pi
        
    return harm_est, spec_cent, perc_est

def ifinf( harm_est, spec_cent, perc_est ):
    if np.isinf(harm_est) == True:
        harm_est = np.pi
    if np.isinf(spec_cent) == True:
        spec_cent = np.pi
    if np.isinf(perc_est) == True:
        perc_est = np.pi
        
    return harm_est, spec_cent, perc_est
"""
def ifnan( value):
    if np.isnan(value) == True:
        value = 2*np.pi    
    return value
def ifinf( value ):
    if np.isinf(value) == True:
        value = np.pi
   
    return value
"""

## These are subject to change as the system matures
def p_proc( subbands, fs ): #(dl, dm, dh, fs):
    """
    For the database initialization...
    Ok not so bad for generalizing this function...
    """
    sources = {}
    for band_num, band in enumerate( subbands ):
        band  = np.nan_to_num( band )
        #band2  = band / np.sqrt(np.sum(band**2))
        #print( 'check: ', np.isfinite(band))

        D = librosa.stft( band )
        H, P = librosa.decompose.hpss(D)
        iyh = librosa.istft(H)
        iyp = librosa.istft(P)

        sources['Band_{}'.format(band_num)] = [ iyh, iyp ]
    
    return sources

def preproc( meas_p, meas_h, sdiv, fs):
    """
    for the input measure...
    """
    subp = subdiv_seg( len(meas_p), sdiv, meas_p)
    subh = subdiv_seg( len(meas_h), sdiv, meas_h)
    
    feature_set = {}
    feature_map = {}
    keys = subp.keys(); print( keys )

    print( len(subh), subh )
    
    
    for ind, j in enumerate( keys ):
        ## When applying to gates remember to put in the pi
        #print('subdiv giving the error: ', j)
        feature_set = {}
        """
        if ind != 0:
            if len(subh[j]) < len(prev):
                temp = subh[j]; print( len(subh[j]), len(prev), len(prev)-len(subh[j]))
                subh[j] = temp + np.zeros(len(prev)-len(subh[j])).tolist()
        """

        harm_est  = 360 / harm(subh[j], fs)
        spec_cent = 360 / spectral_centroid(subh[j], fs) 
        perc_est  = perc( subp[j], fs) / 360 #(2*np.pi) #subp[j], sdiv, 

        harm_est, spec_cent, perc_est = ifnan( harm_est, spec_cent, perc_est )
        harm_est, spec_cent, perc_est = ifinf( harm_est, spec_cent, perc_est )
        
        feature_set['harm_est'] = round(harm_est,3)
        feature_set['spec_cent']= round(spec_cent,3)
        feature_set['perc_est'] = round(perc_est,3)
        
        feature_map[j] = feature_set
        prev = subh[j]
        
    return feature_map

def harm( data, fs ):
    yf = np.fft.fft( data )
    N = data.size
    yf = abs(yf)[:int(len(yf)/2)]

    freq = np.fft.fftfreq(N, 1/fs) * fs/N
    freq = freq[:int(len(freq)/2)]
    #plt.semilogx(freq, yf, '-r'); plt.grid()

    local_maxima  = find_peaks(yf)
    locmx_comp = []
    for i in local_maxima[0]:
        #print(yf[i])
        locmx_comp.append(yf[i])
    loc = locmx_comp
    highest_peak = np.sort(loc)[::-1][:3]

    f_peaks = []; weights = []
    for i in highest_peak:
        ind = np.where( yf == i )
        weights.append( i )
        f_peaks.append(ind[0][0])

    #~~~~~~~ compute weighted average ~~~~~~~~
    f_maxs = [freq[i] for i in f_peaks]
    weights = weights / np.linalg.norm(weights)
    weights = [np.round(w, 3) for w in weights]
    #print(f_maxs); print(weights)
    
    weighted_avg = []
    for index, val in enumerate( weights ):
        w_avg = val * f_maxs[index]
        weighted_avg.append( w_avg )
    
    weighted_avg = np.round( np.sum( weighted_avg ) / np.sum(weights), 3 )
    
    return weighted_avg

def subdiv_seg( bar_dur, sdiv, data ):
        #sdiv will depend on the number of qubits available
        sub_div = bar_dur / 2**sdiv
        subdiv_array = np.zeros(2**sdiv)
        for i, j in enumerate(range(2**sdiv)):
            subdiv_array[i] = int( (j+1) * sub_div )
        
        #~~~~~~~~~~~~ Data Segmentation ~~~~~~~~~~
        
        prev_div = 0
        sub_seg  = {}
        for i, j in enumerate( subdiv_array ):
            key = format(i, '0{}b'.format(sdiv))
            curr_div = int( j )
            sub_seg[key] = ( data[prev_div:curr_div] )
            prev_div = curr_div
            
        return sub_seg

def onset( data, fs ): ## The full audio sample here
    max_amp = np.max( data )
    o_env = librosa.onset.onset_strength(data, sr=fs)
    times = librosa.times_like(o_env, sr=fs)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=fs)
    
    return o_env, max_amp

def perc( data, fs ):
    onset_data, max_amp = onset( data, fs )
    local_maxima  = find_peaks(onset_data)
    #print(local_maxima)
    maxima = [np.round( onset_data, 3 ) for i in local_maxima[0]]
    maxima = np.round( np.sum( maxima ), 3 ) / max_amp
    
    return maxima

def spectral_centroid( data, fs ):
    #print(type(data))
    #data = np.nan_to_num(data)
    #data[data == -inf] = 0.05; data[data == inf] = 0.05
    #data = np.nan_to_num( data ); data = data / np.max(data)
    
    #check = np.isfinite(data); print(check)
   
    """
    
    print('new Track ')
    for count, i in enumerate( check ):
        if i == False:
            print("Ugh>..HERE: ", count, data[count], i)
    
        else:
            print('check: ', data[count], i)
    
    """
    centroid = librosa.feature.spectral_centroid(y=data, sr=fs)
    centroid = round( np.mean(centroid), 3 )
    return centroid

##----------------------------------------------------------------------------------------------
class segmentation:
    """
    Make sure to document how to use this class properly... 
    """
    def __init__( self, sampling_rate, track ):
        self.audio = track
        self.fs = sampling_rate
        self.time_signature = 4

        prior = scipy.stats.uniform(30, 300)
        onset_env = librosa.onset.onset_strength(self.audio, sr=self.fs)
        self.tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.fs, prior=prior)
        self.tempo = round( np.average( self.tempo ), 3 )
        #self.tempo = round( self.tempo[0], 3 )

    def tempo_est( self ):
        return self.tempo

    def measure_segment( self ):
        #Tempo in BPM
        bps = self.tempo / 60
        if ( self.time_signature%4 ) == 0: ## This is hard coded for now...at 4/4
            barline = 4 / bps
            
        barline_samp = int(barline * self.fs)
        return barline, barline_samp
    
    def measure_array( self, bar_samp ):
        measure_array = []
        for index in range( int( len( self.audio ) / bar_samp ) ):
            if index != 0:
                prev_samp = curr_samp
                curr_samp = index * bar_samp
                measure_array.append( self.audio[prev_samp:curr_samp] )
            else:
                curr_samp = 0

        return measure_array
