# https://github.com/kemperlab/Cartan_DMFT/blob/e80c74c2d250af37523f9fa5508246ed016ce89b/src/Step2.py
from functions.processing_funcs import GfromCircuit, getFit, getFitFuncDual, getDualFreq, getSingleFreq, peak_prom
import qiskit
from qiskit import QuantumCircuit, execute, IBMQ
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("..") # Adds higher directory to python modules path.
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, transpile, IBMQ
from qiskit.providers.aer import AerSimulator
from numpy.fft import fft, fftfreq
from numpy import kron
from scipy.signal import find_peaks
from CQS.methods import *
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
#from qiskit.providers.honeywell import Honeywell


from functions.job_circ_funcs import getJobResults, sendJobs, loadMostRecentJob, clearCurrentJob, transpileJobs, loadFullInfo, loadCircuitInfo, saveFullInfo, saveCircuitInfo, getManagedJobResults, sendManagedJobs #chooseFile
from functions.anderson import AIM
from functions.DMFT_funcs import *
from functions.gate_funcs import getGroundStateCircuitFunc, getGroundStateShortCircuitFunc, PauliExpGate, circKHK, GBAt
from functions.circuit_builder_funcs import build_circuit, build_circuit_manual_linear, build_circuit_manual_reduced_k_linear
from numpy import savez

import numpy as np
import scipy
from numpy.fft import fft, fftfreq, fftshift
from scipy.signal import peak_prominences, find_peaks
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt

def highPass(f, cutoffRad, fs, nyquist):
        cuttoffRatio = cutoffRad / nyquist 
        b, a = butter(5, cuttoffRatio, btype='high', analog=False)
        f = filtfilt(b, a, f)
        return f

def toHerz(w):
    return w/(2*np.pi)

def toRad(f):
    return f*(2*np.pi)

def findJobResults(folderName, fs, attemptNumber, Honeywell=False, RetrieveResults = True, PhysicalFiltering = True, MeasurementCorrection = True, NormalizeG = False):
    folderSearch = folderName + 'F_{}'.format(fs)
    print(os.getcwd())
    print(folderSearch)
    print('_v_{}'.format(attemptNumber))
    for path in os.listdir():
        if folderSearch in path:
            if '_v_{}'.format(attemptNumber) in path:
                folder = path
                os.chdir(folder)
                break
            else:
                pass
    if Honeywell:
        MeasurementCorrection = False
    if folder is None:
        print(os.getcwd())
        print(folderSearch)
        print('_v_{}'.format(attemptNumber))
    #Build the Job Information from the pickle file
    times, tNpt, circuitList, Uval, Vstart, sampleFrequency, finalTime, jobID, backendName, filterJobID, shots, state_labels, Greenslist, iterations = loadCircuitInfo('blank')
    
    if not sampleFrequency == fs:
        raise Exception('Mismatched Frequencies: fs = {}, loaded Frequency = {}'.format(fs, sampleFrequency))
    #Get the Job From the provider, IBM or Honeywell
    if not Honeywell:
            IBMQ.load_account()
            #ncsu_provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='phy141')#
            ncsu_provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main') #
            try:
                results = loadMostRecentJob(provider=ncsu_provider, managedJob = True, loadResults=(not RetrieveResults), measurement=MeasurementCorrection)
            except:
                ncsu_provider = IBMQ.get_provider(hub='ibm-q-ncsu', group='nc-state', project='physics-of-spin-')
                try:
                    results = loadMostRecentJob(provider=ncsu_provider, managedJob = True, loadResults=(not RetrieveResults),measurement=MeasurementCorrection)
                except:
                    ncsu_provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='phy141')
                    results = loadMostRecentJob(provider=ncsu_provider, managedJob = True, loadResults=(not RetrieveResults),measurement=MeasurementCorrection)
    else:
        Honeywell.load_account()
        ncsu_provider= Honeywell.get_backend(backendName)
        MeasurementCorrection = False
        results = loadMostRecentJob(provider=ncsu_provider, managedJob = True, loadResults=(not RetrieveResults))
    
    if not Honeywell and MeasurementCorrection:
            measureResults = results[-2]
            #Loading in Results and Processing them
            meas_fitter = CompleteMeasFitter(measureResults, state_labels, circlabel='mcal')
            meas_filter = meas_fitter.filter
    else:
        meas_filter = None
    results = results[-1]
    iGt_imp = np.zeros(tNpt,dtype=complex)
    
    #Get the CIrcuit Results of Green's Function and with the correct filtering
    if MeasurementCorrection:
        iGreal = GfromCircuit(tNpt, results, shots, Greenslist, iGt_imp, readoutCheck=PhysicalFiltering, meas_filter=meas_filter, iterations=iterations)
    else: 
        iGreal = GfromCircuit(tNpt, results, shots, Greenslist, iGt_imp, readoutCheck=PhysicalFiltering, meas_filter=None, iterations=iterations)

    ## Normalize G
    normalFactor = (max([abs(a) for a in iGreal]))
    if NormalizeG:
        iGrealNewNormalized = [i / normalFactor for i in iGreal]
    else:
        iGrealNewNormalized = iGreal
    savez('HighFreq_data', times=times, iGreal=iGreal, Uval=Uval, Vval=Vstart, frequency=fs)

    return iGrealNewNormalized

def extractFrequencyHigh(folderName, fs, N, attemptNumber, expected_w1, Honeywell=False, RetrieveResults = True, PhysicalFiltering = True, MeasurementCorrection = True, NormalizeG = False):
    """
    Args:
        folderName (str): The base component of the data location
        fs (float): The sample Frequency
        N (int): The number of samples
        
    """
    iGreal = findJobResults(folderName, fs, attemptNumber)
    print('Len Results = {}, passed N = {}'.format(len(iGreal),N))
    #Compute the nyquist Frequency for use in the filter
    nyquist = fs/2
    #Manual Hard Filters:
    HighCuttoff = 5 #in Radians
    peakFindCounter = 0
    peaks = []
    while len(peaks) == 0:
        lowPassHerz = min(toHerz(HighCuttoff + peakFindCounter/10), nyquist) - 1e-6
        highPassHerz  = toHerz(2*expected_w1 - peakFindCounter/10)
        w = fftfreq(n=len(iGreal), d=1/fs)
        lowPass = abs(w) < lowPassHerz
        highPass = abs(w) > highPassHerz
        filterCombined = lowPass * highPass
        w = toRad(w[filterCombined])
        # FFT the filtered data, and use average and standard deviation to determine which frequencies are above the cutt off
        # Number of peaks used to determine the quality of the data
        y = abs(fft(iGreal)[filterCombined])
        avg = np.sum(y)/len(y)
        sigma = np.std(y)
        # Uses the Find peaks with a height cuttoff to find the best peaks and how many are above the threshold
        indices = find_peaks(y, height=(avg + 2*sigma), distance = 3)[0]
        for i in range(len(indices)):
            if abs(indices[i] - len(w)/2) < 2: #If the peaks are near the edge of the check region, it just picked up a hump
                indices[i] = 0
        indices = indices[~(indices == 0)]
        peakFindCounter += 1
        if peakFindCounter > 10:
            break
        peaks = w[indices]
        amplitudes = y[indices]
    #Sorts the peaks. Apply [::-1] later to invert the list
    zipped = [(c,w) for c,w in zip(amplitudes,peaks)]
    zipped.sort(key = lambda zipped: zipped[0])
    dw = toRad(fs/N)
    plt.close()
    plt.plot(toRad(fftfreq(n=len(iGreal), d=1/fs)),abs(fft(iGreal)), label='|FFT|')
    plt.plot(w,y, label='Filtered |FFT|')
    plt.legend()
    plt.savefig('HighFreq.png')
    print(zipped[::-1])
    os.chdir('../')
    return zipped[::-1], sigma, avg, iGreal, dw

def extractFrequencyLow(folderName, fs, N, highFreqRad, dw, attemptNumber, expected_max_w1, Honeywell=False, RetrieveResults = True, PhysicalFiltering = True, MeasurementCorrection = True, NormalizeG = False):
    ################
    #Gets the Job From Qiskit
    ################

    iGreal = findJobResults(folderName, fs, attemptNumber)    
    highFreq = toHerz(highFreqRad)
    df = toHerz(dw)
    #Nyquist used for filter
    nyquist = fs/2
    #Apply a filter around the aliased omega_1 frequency
    highPerceived = abs(highFreq - fs*np.rint(highFreq/fs))
    print('Alias Filter Frequency (w):')
    print(toRad(highPerceived))
    """#lowStop = abs((max(highPerceived - 3*df,1e-10))/ nyquist)
    #highStop = abs((min(highPerceived + 3*df, nyquist))/ nyquist) - 1e-6
    #print('Alias Filter Stops (w): {}, {}'.format(toRad(lowStop*nyquist), toRad(highStop*nyquist)))
    #b, a = butter(5, (lowStop, highStop), btype='bandstop', analog=False)
    lowStop = (highPerceived - toHerz(expected_max_w1))/ nyquist
    print('Alias Filter Stop (w): {}'.format(toRad(lowStop*nyquist)))
    b, a = butter(5, lowStop, btype='low', analog=False)
    f = filtfilt(b, a, iGreal)"""
    lowPassHerz =  highPerceived - 2*toHerz(expected_max_w1)
    w = fftfreq(n=len(iGreal), d=1/fs)
    lowpass = abs(w) < lowPassHerz
    w = toRad(w[lowpass])
    #Extract the low frequency from the filtered signal
    y = abs(fft(iGreal)[lowpass])
    avg = np.sum(y)/len(y)
    sigma = np.std(y)
    
    indices = find_peaks((y), distance = 5)[0] #, height=(avg + 2*sigma)
    print('Filtered w values')
    print(w)
    print('Removing "Edge" effects:')
    for i in range(len(indices)):
        print('Peak Index, edge indices')
        print(indices[i], len(w)/2)
        print(abs(indices[i] - len(w)/2) < 4)
        if abs(indices[i] - len(w)/2) < 4: #If the peaks are near the edge of the check region, it just picked up a hump
            indices[i] = 0
    indices = indices[~(indices == 0)]
    peaks = w[indices]
    print(peaks)
    print(y[indices])
    if len(peaks) == 0 or len(peaks) > 4:
        indices = find_peaks((y), height=(avg + sigma), distance = 5)[0]
        for i in range(len(indices)):
            print('Peak Index, edge indices')
            print(indices[i], len(w)/2)
            if abs(indices[i] - len(w)/2) < 4: #If the peaks are near the edge of the check region, it just picked up a hump
                indices[i] = 0
        indices = indices[~(indices == 0)]
        peaks = w[indices]
        if len(peaks) == 0 or len(peaks) > 4:
            indices = find_peaks((y), distance = 5, height=(avg + 2*sigma))[0]
            for i in range(len(indices)):
                if abs(indices[i] - len(w)/2) < 4: #If the peaks are near the edge of the check region, it just picked up a hump
                    indices[i] = 0
            indices = indices[~(indices == 0)]
            peaks = w[indices]
      
    amplitudes = y[indices]
    zipped = [(c,w) for c,w in zip(amplitudes,peaks)]
    zipped.sort(key = lambda zipped: zipped[0])
    dw = toRad(fs/N)
    plt.close()
    plt.plot(toRad(fftfreq(n=len(iGreal), d=1/fs)),abs(fft(iGreal)), label='|FFT|')
    plt.plot(w,y, label='Filtered |FFT|')
    plt.legend()
    plt.savefig('LowFreq.png')
    print(zipped[::-1])
    os.chdir('../')
    return zipped[::-1], sigma, avg, iGreal

