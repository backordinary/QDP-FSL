# https://github.com/stroblme/hqsp-stqft/blob/c2f8f8964648578755d3938bf8658e4c834548e8/qft.py
import time

import datetime

import numpy as np
from numpy import pi

import copy

from math import log2, asin, floor

from qiskit import QuantumRegister, QuantumCircuit, transpile, execute
from qiskit.extensions import UnitaryGate, Initialize
from qiskit.providers.aer.backends.aer_simulator import AerSimulator
from qiskit.providers.aer import Aer, noise
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.library.standard_gates.ry import RYGate, CRYGate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.x import CXGate, XGate


# from qiskit.circuit.library import QFT as qiskit_qft

import inspect
# from qiskit.providers.aer.noise import noise_model
from qiskit.test import mock
from qiskit.providers.exceptions import QiskitBackendNotFoundError
# from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

# from qiskit.circuit.library import QFT

from qiskit.tools.monitor import job_monitor

from frontend import signal
# import mitiq

from utils import PI, filterByThreshold, isPow2
from ibmAccounts import IBMQ#NOT redundant! needed to get account information! Can be commented out if loading e.g. noise data is not needed

def get_bit_string(n, n_qubits):
    """Returns the binary string of an integer with n_qubits characters
    Derived from https://sarangzambare.github.io/jekyll/update/2020/06/13/quantum-frequencies.html
    Args:
        n (int): integer to be converted
        n_qubits (int): number of qubits

    Returns:
        string: binary string
    """
    binaryString = "{0:b}".format(n)
    binaryString = "0"*(n_qubits - len(binaryString)) + binaryString

    return binaryString

def hexKeyToBin(counts, n_qubits):
    """Generates binary representation of a hex based counts
    Derived from https://sarangzambare.github.io/jekyll/update/2020/06/13/quantum-frequencies.html
    Args:
        counts (dict): dictionary with hex keys
        n_qubits (int): number of qubits

    Returns:
        dict: dictionary with bin keys
        n_qubits (int): number of qubits
    """
    out = dict()
    for key, value in counts.items():
        out[format(int(key,16), f'0{int(n_qubits)}b')] = value
    return out, n_qubits

def get_fft_from_counts(counts, n_qubits):
    """Calculates the fft based on the counts of an experiment
    Derived from https://sarangzambare.github.io/jekyll/update/2020/06/13/quantum-frequencies.html
    Args:
        counts (int): dictionary with binary keys
        n_qubits (int): number of qubits

    Returns:
        dict: fft counts
    """
    out = []
    keys = counts.keys()
    for i in range(2**n_qubits):
        id = get_bit_string(i, n_qubits)
        if(id in keys):
            out.append(counts[id])
        else:
            out.append(0)

    return out

def loadBackend(backendName:str, simulation:bool=True, suppressPrint:bool=True):
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider("ibm-q")

    isMock = False
    try:
        backend = provider.get_backend(backendName)
    except QiskitBackendNotFoundError as e:
        print(f"Backend {backendName} not found in {provider}.\nTrying mock backend..")
        
        try:
            tempBackendModule = getattr(mock, backendName.replace("ibmq_", ''))
            backend = inspect.getmembers(tempBackendModule)[0][1]()
            isMock = True
        except QiskitBackendNotFoundError as e:
            print(f"Backend {backendName} also not found in mock devices. Check if the name is valid and has 'ibmq_' as prefix")
        except IndexError:
            print(f"Sorry, but mock backend didn't returned the expected structure. Check {tempBackendModule}")

    if not isMock:
        props = backend.properties(datetime=datetime.datetime.now())
    else:
        props = backend.properties()

    # backend = least_busy(  provider.backends(filters=lambda x: x.configuration().n_qubits >= 5
    #                             and not x.configuration().simulator
    #                             and x.status().operational==True))

    nQubitsAvailable = len(props.qubits)

    try:
        qubitReadoutErrors = [props.qubits[i][4].value for i in range(0, nQubitsAvailable)]
        qubitProbMeas0Prep1 = [props.qubits[i][5].value for i in range(0, nQubitsAvailable)]
        qubitProbMeas1Prep0 = [props.qubits[i][6].value for i in range(0, nQubitsAvailable)]

        if not suppressPrint:
            print(f"Backend {backend} has {nQubitsAvailable} qubits available.")
            print(f"ReadoutErrors are {qubitReadoutErrors}")
            print(f"ProbMeas0Prep1 are {qubitProbMeas0Prep1}")
            print(f"ProbMeas1Prep0 are {qubitProbMeas1Prep0}")
    except IndexError:
        print(f"Failed to get some properties. This can mean that they are simply not stored together with the mock backend")

    if simulation:
        backend = AerSimulator.from_backend(backend)

    return provider, backend

def loadNoiseModel(backendName):
    # set the noise model but do only load the simulator backend. Careful! IBMQ has a request limit ;)
    if type(backendName) == str:
        provider, tempBackend = loadBackend(backendName=backendName, simulation=True)
    else:
        tempBackend = backendName
        provider = None

    # generate noise model from backend properties
    noiseModel = noise.NoiseModel.from_backend(tempBackend)

    return provider, noiseModel

def setupMeasurementFitter( backend, noiseModel, 
                            transpOptLvl, nQubits:int, 
                            nShots:int, nRuns:int=1,
                            suppressPrint:bool=True):
    """In parts taken from https://quantumcomputing.stackexchange.com/questions/10152/mitigating-the-noise-in-a-quantum-circuit

    Args:
        nQubits ([type]): [description]
        nShots (int, optional): [description]. Defaults to 1024.
    """
    # if self.backend is None:
    #     print("Need a backend first")
    #     self.measFitter = None
    #     return None

    # if self.customFilter:
    y = np.ones(2**nQubits)
    ampls = y / np.linalg.norm(y)

    q = QuantumRegister(nQubits,'q')
    qc = QuantumCircuit(q,name="noise mitigation circuit")

    qc.initialize(ampls, [q[i] for i in range(nQubits)])
    qc = qft(qc, nQubits)
    qc.measure_all()

    if noiseModel==None:
        qc = transpile(qc, backend, optimization_level=transpOptLvl) # opt level 0,1..3. 3: heaviest opt
    else:
        qc = transpile(qc, Aer.get_backend('qasm_simulator'), optimization_level=transpOptLvl) # opt level 0,1..3. 3: heaviest opt


    print(f"Running noise measurement {nRuns} times on {nQubits} Qubits with {nShots} shots.. This might take a while")

    jobResults = list()
    for n in range(nRuns):
        job = execute(qc, backend, noise_model=noiseModel, shots=nShots)
        if not suppressPrint:
            job_monitor(job, interval=5) #run a blocking monitor thread
        jobResult = job.result()
        jobResults.append(jobResult.results[0].data.counts)

    filterResultCounts = dict()
    for result in jobResults:
        filterResultCounts = {k: filterResultCounts.get(k, 0) + 1/nRuns*result.get(k, 0) for k in set(filterResultCounts) | set(result)}
    print(f"Filter Results: {filterResultCounts}")

    # else:
    #     measCalibrations, state_labels = complete_meas_cal(qr=QuantumRegister(nQubits), circlabel='mcal')

    #     print(f"Running measurement for filter on {nQubits} Qubits using {nShots} shots")
    #     job = execute(measCalibrations, backend=self.backend, shots=nShots)
    #     job_monitor(job, interval=5)
    #     cal_results = job.result()

    #     self.measFitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    #     print(self.measFitter.cal_matrix)

    # if self.noiseMitigationOpt != 1:
    #     print(f"Enabling noise mitigating option 1 from now on..")
    #     self.noiseMitigationOpt = 1

    return filterResultCounts

# epsilon = 0.12 #enable for error
epsilon = 0 #disable for error
# import random
def qft_rotations(circuit, n, minRotation=0, suppressPrint=True):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    global epsilon

    if n == 0:
        # epsilon += 0.03 #enable for error
        return circuit
    n -= 1
    circuit.h(n) # apply hadamard
    
    rotGateSaveCounter = 0
    # epsilon = 0.04 * random.randint(0,10)
    # epsilon = float(input('epsilon'))
    for qubit in range(n):
        rot = pi/2**(n-qubit) #disable for error
        # rot = pi/2**(n-qubit) + epsilon #enable for error
        if rot <= minRotation:
            rotGateSaveCounter += 1
            if not suppressPrint:
                print(f"Rotations lower than {minRotation}: is {rot}")
        else:
            circuit.cp(rot, qubit, n)

    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    if n != 0 and rotGateSaveCounter != 0 and not suppressPrint:
        print(f"Saved {rotGateSaveCounter} rotation gates which is {int(100*rotGateSaveCounter/n)}% of {n} qubits")
    qft_rotations(circuit, n, minRotation=minRotation, suppressPrint=suppressPrint)
    
def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n, minRotation=0, suppressPrint=False):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n, minRotation=minRotation, suppressPrint=suppressPrint)
    swap_registers(circuit, n)
    # self.measure(circuit,n)
    return circuit

class qft_framework():
    # minRotation = 0.2 #in [0, pi/2)

    def __init__(self,  numOfShots:int=2048,
                        minRotation:int=0, signalThreshold:int=0, fixZeroSignal:bool=False, 
                        suppressPrint:bool=False, draw:bool=False,
                        simulation:bool=True,
                        noiseMitigationOpt:int=0, filterResultCounts=None,
                        useNoiseModel:bool=False, noiseModel=None, backend=None, 
                        transpileOnce:bool=False, transpOptLvl:int=1):
                        
        self.suppressPrint = suppressPrint
        self.numOfShots = numOfShots
        self.minRotation = minRotation
        self.draw = draw

        # check if provided parameters are usefull
        if fixZeroSignal and signalThreshold > 0:
            print("Signal Filter AND zero fixer are enabled. This might result in a wasteful transform. Consider disabling Zero Fixer if not needed.")
        # transfer parameter
        self.fixZeroSignal = fixZeroSignal  
        self.signalThreshold = signalThreshold

        # transfer parameter
        self.transpOptLvl = transpOptLvl      

        # The following code will set:
        # backend
        # provider
        # noiseModel
        # simulation
        # noiseModelBackend
        # -------------------------------------------------

        # no backend -> simulation only
        if backend == None:
            # no simulation without a valid backend doesn't make sense
            if not simulation:
                print("Simulation was disabled but no backend provided. Will enable simulation")
            self.simulation = True
            if useNoiseModel:
                print("Noise model can't be used without a corresponding backend")
            self.noiseModel = None

            self.backend = self.getSimulatorBackend()
            self.provider = None
        # user provided backend only as a name, not instance
        elif type(backend) == str:
            # check if noise model should be used
            if useNoiseModel:
                # check if simulation was disabled
                if not simulation:
                    print("Simulation was disabled but backend provided and noise model enabled. Will enable simulation")
                self.simulation = True

                # if noise model was provided, use it
                if noiseModel!=None:
                    self.noiseModel = noiseModel
                # load it otherwise
                else:
                    # generate noise model from backend properties
                    self.provider, self.noiseModel = loadNoiseModel(backendName=backend)
                self.backend = self.getSimulatorBackend()

            else:
                # check if simulation was enabled
                # if simulation:
                #     print("Simulation was enabled but backend provided and noise model disabled. Will disable simulation")
                # self.simulation = False
                self.simulation = simulation
                # Null the noise model and load a backend for simulation or real device
                self.noiseModel = None
                self.provider, self.backend = loadBackend(backendName=backend, simulation=self.simulation)

        # user provided full backend instance
        else:
            # check if user provided a noise model
            if useNoiseModel:
                if not simulation:
                    print("Simulation was disabled but backend provided and noise model enabled. Will enable simulation")
                self.simulation = True

                # if noise model was provided, use it
                if noiseModel!=None:
                    self.noiseModel = noiseModel
                # load it otherwise
                else:
                    # generate noise model from backend properties
                    self.provider, self.noiseModel = loadNoiseModel(backendName=backend)

                # and set the backend as simulator
                self.backend = self.getSimulatorBackend()
                # and use the backend provided
                # self.backend = backend
            else:
                if simulation:
                    print("Simulation was enabled but backend provided and noise model disabled. Will disable simulation")
                self.simulation = False

                self.noiseModel = None
                self.provider = None #TODO: check if this will cause problems
                self.backend = backend

        # -------------------------------------------------
        
        # transfer parameter
        self.noiseMitigationOpt = noiseMitigationOpt
        self.filterBackend = self.backend
        # # separate backend for noise filter provided?
        # if filterBackend == None:
        #     if self.suppressNoise and self.simulation and useNoiseModel:
        #         print("Warning this might will lead an key error later in transform, as simulation has no noise but noise model was enabled and no filter backend provided")
        #     self.filterBackend = self.backend
        # else:
        #     # check if noise model should be used
        #     if not useNoiseModel:
        #         self.provider, tempBackend = loadBackend(filterBackend, True)
        #         self.filterBackend = tempBackend
        #     else:
        #         self.provider, tempBackend = loadBackend(backend, True)
        #         self.noiseModelBackend = noise.NoiseModel.from_backend(tempBackend)
        #         self.filterBackend = self.getSimulatorBackend()


        # noise mitigation
        self.measFitter = None # used for qiskit noise filter

        # use provided filter resultcounts if possible
        self.filterResultCounts = filterResultCounts # used for custom noise filter
        self.customFilter = True    #TODO: rework such that we can choose a mitigation approach

        # transpilation reuse
        self.transpileOnce = transpileOnce
        self.transpiled = False
        

    def getBackend(self):
        """returns the current backend

        Returns:
            qiskit backend: backend used in qft
        """
        return self.backend

    def getSimulatorBackend(self):
        return Aer.get_backend('qasm_simulator')

    # def setBackend(self, backendName=None, simulation=True):
    #     if backendName != None:
    #         self.provider, self.backend = loadBackend(backendName=backendName, simulation=simulation)
    #     else:
    #         if not self.simulation:
    #             print("Setting simulation to 'True', as no backend was specified")
    #             self.simulation = True
    #         self.backend = self.getSimulatorBackend()

    def estimateSize(self, y_signal:signal):
        assert isPow2(y_signal.nSamples)

        n_qubits = int((log2(y_signal.nSamples)/log2(2)))

        return 2**n_qubits

    def transform(self, y_signal:signal):
        """Apply QFT on a given Signal

        Args:
            y (signal): signal to be transformed

        Returns:
            signal: transformeed signal
        """
        self.samplingRate = y_signal.samplingRate
        y = y_signal.sample()

        if self.signalThreshold > 0:
            # rm when eval done
            y = filterByThreshold(y, self.signalThreshold)

        y_hat = self.processQFT(y)

        return y_hat

    def transformInv(self, y_signal:signal):
        """Apply QFT on a given Signal

        Args:
            y (signal): signal to be transformed

        Returns:
            signal: transformeed signal
        """
        self.samplingRate = y_signal.samplingRate
        y_hat = y_signal.sample()

        y = self.processIQFT(y_hat)

        return y

    def loadMeasurementFitter(self, measFitter):
        self.measFitter = measFitter
        print(self.measFitter.cal_matrix)

        if self.noiseMitigationOpt != 1:
            print(f"Enabling noise mitigating option 1 from now on..")
            self.noiseMitigationOpt = 1

    

    def qubitNoiseFilter(self, jobResult, nQubits:int):
        """In parts taken from https://quantumcomputing.stackexchange.com/questions/10152/mitigating-the-noise-in-a-quantum-circuit

        Args:
            jobResult ([type]): [description]

        Returns:
            [type]: [description]
        """
        # if self.customFilter:
        if self.filterResultCounts == None:
            print("Need to initialize measurement fitter first")
            if nQubits == None:
                print(f"For auto-initialization, you must provide the number of qubits")
                return jobResult
            self.filterResultCounts=setupMeasurementFitter( backend=self.filterBackend, noiseModel=self.noiseModel,
                                                            transpOptLvl=self.transpOptLvl, nQubits=nQubits, 
                                                            nShots=jobResult.results[0].shots)
        elif len(self.filterResultCounts) == 1:
            print("Seems like you try to mitigate noise of a simulation without any noise. You can either disable noise suppression or consider running with noise.")
            return jobResult
        
        
        mitigatedResult = copy.deepcopy(jobResult)

        jobResultCounts = jobResult.results[0].data.counts

        maxCount = max(jobResultCounts.values()) #get max. number of counts in the plot

        nMitigated=0
        for idx, count in jobResultCounts.items():
            if count/maxCount < 0.5 or idx == "0x0":    # only filter counts which are less than half of the chance
                # pretty complicated line, but we are converting just from hex indexing to binary here and padding zeros where necessary
                # filterResultCounts[bin_zero_padded]: idx:hex -> bin -> bin zero padded 
                # mitigatedResult.results[0].data.counts[idx] = max(0,count - self.filterResultCounts[format(int(idx,16), f'0{int(log2(nQubits))}b')])
                if idx in self.filterResultCounts:
                    mitigatedResult.results[0].data.counts[idx] = max(0,count - self.filterResultCounts[idx])
                    nMitigated+=1
                # it can (and often will) happen, that the result list contains keys which are not in the filter result counts
                # especially in large circuits, this is the case, as there are so many computational basis states (2**nQubits)
                # that it's very unlikely every state is covered by just an initialized circuit (like the filter)

        if not self.suppressPrint:
            print(f"Mitigated {nMitigated} in total")

        # else:
            # if self.measFitter == None:
            #     print("Need to initialize measurement fitter first")
            #     if nQubits == None:
            #         print(f"For auto-initialization, you must provide the number of qubits")
            #         return jobResult
            #     self.setupMeasurementFitter(nQubits=nQubits)

            # # Get the filter object
            # measFilter = self.measFitter.filter

            # # Results with mitigation
            # mitigatedResult = measFilter.apply(jobResult)
            # # mitigatedCounts = mitigatedResult.get_counts(0)
            # print(f"Filtering achieved at '0000': {mitigatedResult.get_counts()['0000']} vs before: {jobResult.get_counts()['0000']}")
        return mitigatedResult

    def mitiqNoiseFilter(self, jobResult, nQubits:int):
        return jobResult

    def showCircuit(self, y):
        """Display the circuit for a signal y

        Args:
            y (signal): signal instance used for circuit configuration
        """
        self.transform(y,int(y.size / 3))


    

    

    def measure(self, circuit, n):
        for qubit in range(n):
            circuit.barrier(qubit)
        circuit.measure_all()



    def inverseQft(self, circuit, n):
        """Inverse QFT on the first n qubits in the circuit"""
        q_circuit = qft(QuantumCircuit(n), n)
        inv_q_ciruit = q_circuit.inverse()
        circuit.append(inv_q_ciruit, circuit.qubits[:n])

        return circuit.decompose()


    def preprocessSignal(self, y, scaler, shift=False):
        '''
        Preprocessing signal using a provided scaler
        '''

        y = y*scaler
        if shift:
            y = y + abs(min(y))

        return y

    def dense(self, y_hat, D=3):
        if D==0:
            return y_hat
        D = int(D)
        y_hat_densed = np.zeros(int(y_hat.size/D))

        for i in range(y_hat_densed.size*D):
            # if i%D != 0:
            y_hat_densed[int(i/D)] += abs(y_hat[i])

        return y_hat_densed

    def gram_schmidt(self,A):
    
        (n, m) = A.shape
        
        for i in range(m):
            
            q = A[:, i] # i-th column of A
            
            for j in range(i):
                q = q - np.dot(A[:, j], A[:, i]) * A[:, j]
            
            if np.array_equal(q, np.zeros(q.shape)):
                raise np.linalg.LinAlgError("The column vectors are not linearly independent")
            
            # normalize q
            q = q / np.sqrt(np.dot(q, q))
            
            # write the vector back in the matrix
            A[:, i] = q

        return A

    def _multiplex(self, target_gate, list_of_angles, last_cnot=True):
        """
        Return a recursive implementation of a multiplexor circuit,
        where each instruction itself has a decomposition based on
        smaller multiplexors.

        The LSB is the multiplexor "data" and the other bits are multiplexor "select".

        Args:
            target_gate (Gate): Ry or Rz gate to apply to target qubit, multiplexed
                over all other "select" qubits
            list_of_angles (list[float]): list of rotation angles to apply Ry and Rz
            last_cnot (bool): add the last cnot if last_cnot = True

        Returns:
            DAGCircuit: the circuit implementing the multiplexor's action
        """
        list_len = len(list_of_angles)
        local_num_qubits = int(log2(list_len)) + 1

        q = QuantumRegister(local_num_qubits)
        circuit = QuantumCircuit(q, name="multiplex" + local_num_qubits.__str__())

        lsb = q[0]
        msb = q[local_num_qubits - 1]

        # case of no multiplexing: base case for recursion
        if local_num_qubits == 1:
            circuit.append(target_gate(list_of_angles[0]), [q[0]])
            return circuit

        # calc angle weights, assuming recursion (that is the lower-level
        # requested angles have been correctly implemented by recursion
        angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]], np.identity(2 ** (local_num_qubits - 2)))

        # calc the combo angles
        list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()

        # recursive step on half the angles fulfilling the above assumption
        multiplex_1 = self._multiplex(target_gate, list_of_angles[0 : (list_len // 2)], False)
        circuit.append(multiplex_1.to_instruction(), q[0:-1])

        # attach CNOT as follows, thereby flipping the LSB qubit
        circuit.append(CXGate(), [msb, lsb])

        # implement extra efficiency from the paper of cancelling adjacent
        # CNOTs (by leaving out last CNOT and reversing (NOT inverting) the
        # second lower-level multiplex)
        multiplex_2 = self._multiplex(target_gate, list_of_angles[(list_len // 2) :], False)
        if list_len > 1:
            circuit.append(multiplex_2.to_instruction().reverse_ops(), q[0:-1])
        else:
            circuit.append(multiplex_2.to_instruction(), q[0:-1])

        # attach a final CNOT
        if last_cnot:
            circuit.append(CXGate(), [msb, lsb])

        return circuit

    def qiskitDebugInitialize(self, y:np.array, nQubits:int, circuit:QuantumCircuit, registers:QuantumRegister):
        circuit.reset([registers[i] for i in range(nQubits)])

        # ident = np.identity(2**nQubits)
        # y += 0.0004
        # np.fill_diagonal(ident, y)
        # ident = self.gram_schmidt(ident)
        # uEncoded = UnitaryGate(ident)
        # circuit.append(uEncoded, [registers[i] for i in range(nQubits)])

        q = QuantumRegister(nQubits)
        disentangling_circuit = QuantumCircuit(q, name="disentangler")

        # kick start the peeling loop, and disentangle one-by-one from LSB to MSB
        remaining_param = y

        for i in range(nQubits):
            # work out which rotations must be done to disentangle the LSB
            # qubit (we peel away one qubit at a time)
            (remaining_param, thetas, phis) = Initialize._rotations_to_disentangle(remaining_param)

            # perform the required rotations to decouple the LSB qubit (so that
            # it can be "factored" out, leaving a shorter amplitude vector to peel away)

            add_last_cnot = True
            if np.linalg.norm(phis) != 0 and np.linalg.norm(thetas) != 0:
                add_last_cnot = False

            if np.linalg.norm(phis) != 0:
                rz_mult = self._multiplex(RZGate, phis, last_cnot=add_last_cnot)
                disentangling_circuit.append(rz_mult.to_instruction(), q[i : nQubits])

            if np.linalg.norm(thetas) != 0:
                ry_mult = self._multiplex(RYGate, thetas, last_cnot=add_last_cnot)
                disentangling_circuit.append(ry_mult.to_instruction().reverse_ops(), q[i : nQubits])
        disentangling_circuit.global_phase -= np.angle(sum(remaining_param))


        initialize_instr = disentangling_circuit.to_instruction().inverse()

        q = QuantumRegister(self.num_qubits, "q")
        circuit = QuantumCircuit(q, name="init_def")
        for qubit in q:
            circuit.append(Reset(), [qubit])
        circuit.append(initialize_instr, q[:])

    def gen_angles(self, y:np.array):
        angles = list()
        if len(y) > 1:
            new_y = list()
            for k in range(int(len(y)/2)):
                new_y.append(np.sqrt(y[2*k]**2 + y[2*k+1]**2))
            inner_angles = self.gen_angles(new_y)

            for k in range(len(new_y)):
                if new_y[k] != 0:
                    if y[2*k] > 0:
                        angles.append(2*asin((y[2*k+1])/new_y[k]))
                    else:
                        angles.append(2*pi-2*asin((y[2*k+1])/new_y[k]))
                else:
                    angles.append(0)
        return angles

    def gen_angles_z(self, y:np.array):
        angles_z = list()

        if len(y) > 1:
            new_y = list()
            for k in range(int(len(y)/2)):
                new_y.append((y[2*k] + y[2*k+1])/2)
            inner_angles_z = self.gen_angles(new_y)

            for k in range(len(new_y)):
                angles_z.append(y[2*k+1]-y[2*k])

            angles_z = inner_angles_z + angles_z

    def gen_circuit(self, angles:np.array, nQubits:int, circuit:QuantumCircuit, registers:QuantumRegister):
        for k in range(len(angles)-2):
            if k==0:
                circuit.ry(angles[k], registers[k])
            else:
                if angles[k] > self.minRotation:
                    circuit.mcry(angles[k], [registers[i] for i in range(max(floor(log2(k)),1))], registers[max(floor(log2(k)),1)])
        
        return circuit

    def gen_circuit_fast(self, angles:np.array, angles_z:np.array, nQubits:int, circuit:QuantumCircuit, registers:QuantumRegister):
        for k in range(len(angles)-2):
            if angles[k] != 0:
                circuit.ry(angles[k], registers[log2(k)])

        for k in range(len(angles_z)-2):
            if angles_z[k] != 0:
                circuit.ry(angles_z[k], registers[log2(k)])

        


    def encoding(self, y:np.array, nQubits:int, circuit:QuantumCircuit, registers:QuantumRegister, customInitialize:bool=False):
        if customInitialize:
            circuit.reset([registers[i] for i in range(nQubits)])
            angles=self.gen_angles(y)
            # angles_z=self.gen_angles_z(y)
            # circuit = self.gen_circuit_fast(angles=angles, angles_z=angles_z, nQubits=nQubits, circuit=circuit, registers=registers)
            circuit = self.gen_circuit(angles=angles, nQubits=nQubits, circuit=circuit, registers=registers)

        else:
            circuit.initialize(y, [registers[i] for i in range(nQubits)])

        return circuit

    def processQFT(self, y:np.array):
        n_samples = y.size
        assert isPow2(n_samples)

        nQubits = int((log2(n_samples)/log2(2)))
        if not self.suppressPrint:
            print(f"Using {nQubits} Qubits to encode {n_samples} Samples")     

        if y.max() == 0.0:
            if self.fixZeroSignal:
                print(f"Warning: Signal's max value is zero and therefore amplitude initialization will fail. Setting signal to constant-one to continue")
                y = np.ones(n_samples)
            else:
                if not self.suppressPrint:
                    print(f"Zero Signal and fix should not be applied. Will return zero signal with expected length")
                y_hat = np.zeros(2**nQubits)
                return y_hat

        # Normalize ampl, which is required for squared sum of amps=1
        ampls = y / np.linalg.norm(y)
        q = QuantumRegister(nQubits,'q')

        # transpile once enabled but not transpiled yet -> run full process
        if self.transpileOnce and not self.transpiled:
            # setup the transpiled circuit storage for the generic qft circuit
            self.transpiledQ = QuantumRegister(nQubits,'q')
            self.transpiledQC = QuantumCircuit(self.transpiledQ)
            self.transpiledQC = qft(self.transpiledQC, nQubits, minRotation=self.minRotation, suppressPrint=self.suppressPrint)
            self.transpiledQC.measure_all()
    

            if not self.suppressPrint:
                print(f"Transpiling for {self.backend}")
            if not self.suppressPrint:
                print(f"QFT Depth before transpiling: {self.transpiledQC.depth()}")

            # do the transpilation of the qft
            self.transpiledQC = transpile(self.transpiledQC, self.backend, optimization_level=self.transpOptLvl) # opt level 0,1..3. 3: heaviest opt

            if not self.suppressPrint:
                print(f"QFT Depth after transpiling: {self.transpiledQC.depth()}")

                
            # initialize the "first" layer
            qc = QuantumCircuit(q, name="qft circuit")
            # qc.initialize(ampls, [q[i] for i in range(nQubits)])
            qc = self.encoding(y=ampls, nQubits=nQubits, circuit=qc, registers=q)

            if not self.suppressPrint:
                print(f"Starting transpile for initialization circuit")

            # do the transpilation of the encoding layer
            qc = transpile(qc, self.backend, optimization_level=self.transpOptLvl)

            # append the transpiled qft circuit to the encoding layer
            qc = qc + self.transpiledQC

            if not self.suppressPrint:
                print(f"Transpile once enabled, so next time we only need to transpile the encoding layer")
            # set transpiled flag
            self.transpiled = True

        # transpile once enabled and qft circuit already transpiled -> only encoding layer needs to be transpiled
        elif self.transpileOnce and self.transpiled:
            # initialize the "first" layer
            qc = QuantumCircuit(q, name="qft circuit")
            # qc.initialize(ampls, [q[i] for i in range(nQubits)])
            qc = self.encoding(y=ampls, nQubits=nQubits, circuit=qc, registers=q)

            # do the transpilation of the encoding layer
            qc = transpile(qc, self.backend, optimization_level=self.transpOptLvl)

            # append the transpiled qft circuit to the encoding layer
            qc = qc + self.transpiledQC

        # transpile once disabled -> regular process but no seperate transpilation for encoding and qft
        else:
            qc = QuantumCircuit(q, name="qft circuit")

            # for 2^n amplitudes, we have n qubits for initialization
            # this means that the binary representation happens exactly here
            qc.initialize(ampls, [q[i] for i in range(nQubits)])
            qc = qft(qc, nQubits, minRotation=self.minRotation, suppressPrint=self.suppressPrint)
            qc.measure_all()
            qc = transpile(qc, self.backend, optimization_level=self.transpOptLvl) # opt level 0,1..3. 3: heaviest opt

        if self.draw:
            self.draw=False
            name = str(time.mktime(datetime.datetime.now().timetuple()))[:-2]
            # qc.draw(output='mpl', filename=f'./export/{name}.png')
            MAIN='#06574b'
            WHITE='#FFFFFF'
            GRAY='#BBBBBB'
            HIGHLIGHT='#9202e1'
            LIGHTGRAY='#EEEEEE'
            qc.draw(output='mpl', filename=f"./harmonicSignal_qftCircuit.pdf", fold=-1, style={'displaycolor': {  'cp': (MAIN, WHITE),
                                                                                                                    'x': (MAIN, WHITE),
                                                                                                                    'measure': (MAIN, WHITE),
                                                                                                                    'initialize': (MAIN, WHITE),
                                                                                                                    'swap': (MAIN, WHITE),
                                                                                                                    'h': (MAIN, WHITE)}})

        


        # qc = transpile(qc, self.backend, optimization_level=self.transpOptLvl) # opt level 0,1..3. 3: heaviest opt


        if not self.suppressPrint:
            start = time.time()
            print(f"Executing job at {start}...")
    
        #substitute with the desired backend
        job = execute(qc, self.backend,shots=self.numOfShots,noise_model=self.noiseModel)
        # if job.status != "COMPLETED":
        if not self.suppressPrint:
            job_monitor(job, interval=5) #run a blocking monitor thread

        # self.lastJobResultCounts = job.result().get_counts()
        
        if not self.suppressPrint:
            end = time.time()
            print(f"Finished at {end}; took {end-start}")
        if not self.suppressPrint:
            print("Post Processing...")
        
        if self.noiseMitigationOpt == 1:
            jobResult = self.qubitNoiseFilter(jobResult=job.result(), nQubits=nQubits)
        elif self.noiseMitigationOpt == 2:
            jobResult = self.mitiqNoiseFilter(jobResult=job.result(), nQubits=nQubits)
        else:
            if not self.suppressPrint:
                print("Warning: Mitigating results is implicitly disabled. Consider enabling it by running 'setupMeasurementFitter'")
            jobResult = job.result()

        counts = jobResult.get_counts()
        y_hat = np.array(get_fft_from_counts(counts, nQubits))

        # [:n_samples//2]
        # y_hat = self.dense(y_hat, D=max(n_qubits/(self.samplingRate/n_samples),1))

        # Omitting normalization here, since we normalize in post
        y_hat = y_hat * 1/self.numOfShots
        # y_hat = y_hat*(1/y_hat.max())

        # top_indices = np.argsort(-np.array(fft))
        # freqs = top_indices*self.samplingRate/n_samples
        # get top 5 detected frequencies

        
        return y_hat

    # def executor(circuit: mitiq.QPROGRAM) -> float:
    #     pass

    def processIQFT(self, y:np.array):
        n_samples = y.size
        assert isPow2(n_samples)

        n_qubits = int((log2(n_samples)/log2(2)))
        if not self.suppressPrint:
            print(f"Using {n_qubits} Qubits to encode {n_samples} Samples")     

        if y.max() == 0.0:
            y_hat = np.zeros(2**n_qubits)
            return y_hat

        q = QuantumRegister(n_qubits)
        qc = QuantumCircuit(q)

        # Normalize ampl, which is required for squared sum of amps=1
        ampls = y / np.linalg.norm(y)

        # for 2^n amplitudes, we have n qubits for initialization
        # this means that the binary representation happens exactly here
        qc.initialize(ampls, [q[i] for i in range(n_qubits)])

        qc = qft(qc, n_qubits)
        qc.inverse()
        qc.measure_all()

        qasm_backend = Aer.get_backend('qasm_simulator')
        # real_backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 5
        #                                        and not x.configuration().simulator
        #                                        and x.status().operational==True))


        #substitute with the desired backend
        out = execute(qc, qasm_backend, shots=self.numOfShots).result()
        counts = out.get_counts()
        y_hat = np.array(get_fft_from_counts(counts, n_qubits))
        # [:n_samples//2]
        y_hat = self.dense(y_hat, D=max(n_qubits/(self.samplingRate/n_samples),1))
        # top_indices = np.argsort(-np.array(fft))
        # freqs = top_indices*self.samplingRate/n_samples
        # get top 5 detected frequencies

        if self.draw:
            self.draw=False
            name = str(time.mktime(datetime.datetime.now().timetuple()))[:-2]
            qc.draw(output='mpl', filename=f'./export/{name}.png')
        return y_hat

