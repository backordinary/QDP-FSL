# https://github.com/h-rathee851/Pulse_application_qiskit/blob/62d2cd576bafffcee1ef6204bcab3de943231f27/src/pulsecalibration.py
"""
Object to calibrate pulse of backend and qubit of interest.
"""

# Importing required python packages
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Importing standard Qiskit libraries
from qiskit import IBMQ, execute, pulse
from qiskit.providers.ibmq import IBMQBackend
from qiskit.pulse import DriveChannel, Schedule, Play
from qiskit.pulse import library as pulse_lib
from qiskit.pulse.library import Waveform, Gaussian
from qiskit.tools.monitor import job_monitor
from qiskit.providers.exceptions import QiskitBackendNotFoundError, BackendConfigurationError

# Loading your IBM Q account(s)
#IBMQ.load_account()
#provider = IBMQ.get_provider()


class PulseCalibration():
    """Creates an object that is used for pulse calibration.

    Args:
        backend (IBMQBackend) : The IBMQBackend for which pulse calibration needs to be done.
        qubit (int) : The qubit for which the pulse calibration is done.
        qubit_freq_ground (float) : Custom frequency for 0->1 transition.
        qubit_freq_excited (float) : Custom frequency for 1->2 transition.
        pi_amp_ground (float) : Custom pi amplitude for 0->1 transition.
                                The value should be between 0 and 1.
        pi_amp_excited (float) : Custom pi amplitude for 1->2 transition.
                                 The value should be between 0 and 1.
        pulse_dur (int) : The duration of the pi pulse to be used for calibration.
        pulse_sigma (int) : The standard deviation of the pi pulse to be used for calibration.

    """

    def __init__(self, backend, qubit, qubit_freq_ground=None, qubit_freq_excited=None,
                 pi_amp_ground=None, pi_amp_excited=None, pulse_dur=None, pulse_sigma=None):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments
        if not isinstance(backend, IBMQBackend):
            raise QiskitBackendNotFoundError("Provided backend not available." +
                                             "Please provide backend after obtaining from IBMQ.")
        self._backend = backend
        self._back_config = backend.configuration()
        if qubit >= self._back_config.n_qubits:
            raise BackendConfigurationError(f"Qubit {qubit} not present in the provided backend.")
        self._qubit = qubit
        self._back_defaults = backend.defaults()
        self._dt = self._back_config.dt
        self._qubit_freq = self._back_defaults.qubit_freq_est[self._qubit]
        self._inst_sched_map = self._back_defaults.instruction_schedule_map
        self._drive_chan = DriveChannel(qubit)
        if pulse_sigma:
            self._pulse_sigma = pulse_sigma
        else:
            self._pulse_sigma = 40
        if pulse_dur:
            self._pulse_duration = pulse_dur
        else:
            self._pulse_duration = (4*self._pulse_sigma)-((4*self._pulse_sigma) % 16)
        self._qubit_freq_ground = qubit_freq_ground
        self._qubit_freq_excited = qubit_freq_excited
        self._pi_amp_ground = pi_amp_ground
        self._pi_amp_excited = pi_amp_excited
        self._state_discriminator_012 = None
        # Find out which measurement map index is needed for this qubit
        meas_map_idx = None
        for i, measure_group in enumerate(self._back_config.meas_map):
            if qubit in measure_group:
                meas_map_idx = i
                break
        assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"
        # The measurement pulse for measuring the qubit of interest.
        self._measure = self._inst_sched_map.get('measure',
                                                 qubits=self._back_config.meas_map[meas_map_idx])

    def create_cal_circuit(self, amp):
        """Constructs and returns a schedule containing Gaussian pulse with amplitude as 'amp',
           sigma as 'pulse_sigma' and duration as 'pulse_duration'."""
        sched = Schedule()
        sched += Play(Gaussian(duration=self._pulse_duration, sigma=self._pulse_sigma,
                               amp=amp), self._drive_chan)
        sched += self._measure << sched.duration
        return sched

    def create_cal_circuit_excited(self, base_pulse, freq):
        """ Constructs and returns a schedule containing a pi pulse for 0->1 transition followed by
            a sidebanded pulse which corresponds to applying 'base_pulse' at frequency 'freq'."""
        sched = Schedule()
        sched += Play(Gaussian(duration=self._pulse_duration, sigma=self._pulse_sigma,
                               amp=self._pi_amp_ground), self._drive_chan)
        sched += Play(self.apply_sideband(base_pulse, freq), self._drive_chan)
        sched += self._measure << sched.duration
        return sched

    @staticmethod
    def _fit_function(x_values, y_values, function, init_params):
        """ A function fitter. Returns the fit parameters of 'function'."""
        fitparams, _ = curve_fit(function, x_values, y_values, init_params)
        y_fit = function(x_values, *fitparams)
        return fitparams, y_fit

    @staticmethod
    def _baseline_remove(values):
        """Centering data around zero."""
        return np.array(values) - np.mean(values)

    def apply_sideband(self, pulse, frequency):
        """Apply a sine sideband to 'pulse' at frequency 'freq'.
        Args:
            pulse (Waveform): The pulse to which sidebanding is to be applied.
            frequency (float): LO frequency at which the pulse is to be applied.

        Returns:
            Waveform: The sidebanded pulse.
        """
        t_samples = np.linspace(0, self._dt*self._pulse_duration, self._pulse_duration)
        sine_pulse = np.sin(2*np.pi*(frequency-self._qubit_freq_ground)*t_samples)
        sideband_pulse = Waveform(np.multiply(np.real(pulse.samples), sine_pulse),
                                  name='sideband_pulse')
        return sideband_pulse

    def get_job_data(self, job, average):
        """Retrieve data from a job that has already run.
        Args:
            job (Job): The job whose data you want.
            average (bool): If True, gets the data assuming data is an average.
                            If False, gets the data assuming it is for single shots.
        Return:
            list: List containing job result data.
        """
        scale_factor = 1e-14
        job_results = job.result(timeout=120)  # timeout parameter set to 120 s
        result_data = []
        for i in range(len(job_results.results)):
            if average:  # get avg data
                result_data.append(job_results.get_memory(i)[self._qubit]*scale_factor)
            else:  # get single data
                result_data.append(job_results.get_memory(i)[:, self._qubit]*scale_factor)
        return result_data

    # Prints out relative maxima frequencies in output_data; height gives lower bound (abs val)
    @staticmethod
    def _rel_maxima(freqs, output_data, height, distance):
        """Prints out relative maxima frequencies in output_data (can see peaks);
           height gives upper bound (abs val). Be sure to set the height properly or
           the peak will be ignored!
        Args:
            freqs (list): frequency list
            output_data (list): list of resulting signals
            height (float): upper bound (abs val) on a peak
            width (float): 
        Returns:
            list: List containing relative maxima frequencies
        """
        peaks, _ = find_peaks(x=output_data, height=height, distance=distance)
        return freqs[peaks]

    def find_freq_ground(self, verbose=False, visual=False):
        """Sets and returns the calibrated frequency corresponding to 0->1 transition."""
        # pylint: disable=too-many-locals
        sched_list = [self.create_cal_circuit(0.5)]*75
        freq_list = np.linspace(self._qubit_freq-(40*1e+6), self._qubit_freq+(40*1e+6), 75)
        sweep_job = execute(sched_list, backend=self._backend, meas_level=1, meas_return='avg',
                            shots=1024, schedule_los = [{self._drive_chan: freq} for freq in freq_list])

        if verbose:
            print("Executing the Frequency sweep job for 0->1 transition.")
            job_monitor(sweep_job)
        sweep_result = sweep_job.result()
        sweep_values = []
        for i in range(len(sweep_result.results)):
            # Get the results from the ith experiment
            res = sweep_result.get_memory(i)*1e-14
            # Get the results for `qubit` from this experiment
            sweep_values.append(res[self._qubit])

        freq_list_GHz = freq_list/1e+9
        if visual:
            print("The frequency-signal plot for frequency sweep: ")
            plt.scatter(freq_list_GHz, np.real(sweep_values), color='black')
            plt.xlim([min(freq_list_GHz), max(freq_list_GHz)])
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("Measured Signal [a.u.]")
            plt.show()

        def find_init_params(res_values):
            est_baseline=np.mean(res_values)
            est_slope=-5 if est_baseline-np.min(res_values)> 2 else 5
            return [est_slope, self._qubit_freq/1e9, 1, est_baseline]
        init_params = find_init_params(np.real(sweep_values))
        # Obtain the optimal paramters that fit the result data.
        def lorentzian(xval, scale, q_freq, hwhm, shift):
            return (scale / np.pi) * (hwhm / ((xval - q_freq)**2 + hwhm**2)) + shift
        fit_params, y_fit = self._fit_function(freq_list_GHz,
                                               np.real(sweep_values),
                                               lorentzian,
                                               init_params  # init parameters for curve_fit
                                               )
        if visual:
            print("The frequency-signal plot for frequency sweep: ")
            plt.scatter(freq_list_GHz, np.real(sweep_values), color='black')
            plt.plot(freq_list_GHz, y_fit, color='red')
            plt.xlim([min(freq_list_GHz), max(freq_list_GHz)])
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("Measured Signal [a.u.]")
            plt.show()
        _, qubit_freq_new, _, _ = fit_params
        self._qubit_freq_ground = qubit_freq_new*1e9
        if verbose:
            print(f"The calibrate frequency for the 0->1 transition is {self._qubit_freq_ground}")
        return self._qubit_freq_ground

    def find_pi_amp_ground(self, verbose=False, visual=False):
        """Sets and returns the amplitude of the pi pulse corresponding to 0->1 transition."""
        # pylint: disable=too-many-locals
        if not self._qubit_freq_ground:
            warn("ground_qubit_freq not computed yet and custom qubit freq not provided." +
                 "Computing ground_qubit_freq now.")
            self._qubit_freq_ground = self.find_freq_ground(verbose, visual)
        amp_list = np.linspace(0, 0.74, 75)
        rabi_sched_list = [self.create_cal_circuit(amp) for amp in amp_list]
        rabi_list_len = len(rabi_sched_list)
        rabi_job = execute(rabi_sched_list,backend=self._backend, meas_level=1, meas_return='avg', 
                           shots=1024, schedule_los=[{self._drive_chan: self._qubit_freq_ground}]*len(rabi_sched_list))
        if verbose:
            print("Executing the rabi oscillation job to get Pi pulse for 0->1 transition.")
            job_monitor(rabi_job)
        rabi_results = rabi_job.result()
        scale_factor = 1e-14
        rabi_values = []
        for i in range(75):
            # Get the results for `qubit` from the ith experiment
            rabi_values.append(rabi_results.get_memory(i)[self._qubit]*scale_factor)

        rabi_values = np.real(self._baseline_remove(rabi_values))

        # Obtain the optimal paramters that fit the result data.
        fit_params, y_fit = self._fit_function(amp_list,
                                               rabi_values,
                                               lambda x, A, B, drive_period, phi: (A*np.sin(2*np.pi*x/drive_period - phi) + B),
                                               [3e-7, 0, 0.3, 0])
        drive_period = fit_params[2]
        self._pi_amp_ground = drive_period/2
        if verbose:
            print(f"The Pi amplitude of 0->1 transition is {self._pi_amp_ground}.")
        if visual:
            print("The amplitude-signal plot for rabi oscillation for 0->1 transition.")
            plt.figure()
            plt.scatter(amp_list, rabi_values, color='black')
            plt.plot(amp_list, y_fit, color='red')
            plt.axvline(drive_period/2, color='red', linestyle='--')
            plt.axvline(drive_period, color='red', linestyle='--')
            plt.annotate("", xy=(drive_period, 0), xytext=(drive_period/2, 0),
                         arrowprops=dict(arrowstyle="<->", color='red'))
            #plt.annotate("$\pi$", xy=(drive_period/2-0.03, 0.1), color='red')

            plt.xlabel("Drive amp [a.u.]", fontsize=15)
            plt.ylabel("Measured signal [a.u.]", fontsize=15)
            plt.show()

        return self._pi_amp_ground

    def find_freq_excited(self, verbose=False, visual=False):
        """Sets and returns the frequency corresponding to 1->2 transition."""
        # pylint: disable=too-many-locals
        if not self._qubit_freq_ground:
            raise ValueError("The qubit_freq_ground is not determined. Please determine" +
                             "qubit_freq_ground first.")
        if not self._pi_amp_ground:
            raise ValueError("The pi_amp_ground is not determined.\
                               Please determine pi_amp_ground first.")
        base_pulse = pulse_lib.gaussian(duration=self._pulse_duration,
                                        sigma=self._pulse_sigma, amp=0.3)
        sched_list = []
        # Here we assume that the anharmocity is about 8% for all qubits.
        excited_freq_list = self._qubit_freq_ground-0.33e9 + np.linspace(-100*1e+6, 100*1e+6, 75)
        for freq in excited_freq_list:
            sched_list.append(self.create_cal_circuit_excited(base_pulse, freq))
        excited_sweep_job = execute(sched_list, backend=self._backend, 
                                    meas_level=1, meas_return='avg', shots=1024,
                                    schedule_los=[{self._drive_chan: self._qubit_freq_ground}]*len(excited_freq_list)
                                    )
        if verbose:
            print("Executing the Frequency sweep job for 1->2 transition.")
            job_monitor(excited_sweep_job)
        excited_sweep_data = self.get_job_data(excited_sweep_job, average=True)
        if visual:
            print("The frequency-signal plot of frequency sweep for 1->2 transition.")
            # Note: we are only plotting the real part of the signal
            plt.scatter(excited_freq_list/1e+9, excited_sweep_data, color='black')
            plt.xlim([min(excited_freq_list/1e+9), max(excited_freq_list/1e+9)])
            plt.xlabel("Frequency [GHz]", fontsize=15)
            plt.ylabel("Measured Signal [a.u.]", fontsize=15)
            plt.title("1->2 Frequency Sweep", fontsize=15)
            plt.show()
        def lorentzian(xval, scale, q_freq, hwhm, shift):
            return (scale / np.pi) * (hwhm / ((xval - q_freq)**2 + hwhm**2)) + shift
        # do fit in Hz
        excited_sweep_fit_params, excited_sweep_y_fit = self._fit_function(excited_freq_list,
                                                   excited_sweep_data,
                                                   lorentzian,
                                                   [12, self._qubit_freq_ground*0.92, 0.05*1e+9, 1e-7]
                                                   )
        if visual:
            print("The frequency-signal plot of frequency sweep for 1->2 transition.")
            # Note: we are only plotting the real part of the signal
            plt.scatter(excited_freq_list/1e+9, excited_sweep_data, color='black')
            plt.plot(excited_freq_list/1e+9, excited_sweep_y_fit, color='red')
            plt.xlim([min(excited_freq_list/1e+9), max(excited_freq_list/1e+9)])
            plt.xlabel("Frequency [GHz]", fontsize=15)
            plt.ylabel("Measured Signal [a.u.]", fontsize=15)
            plt.title("1->2 Frequency Sweep", fontsize=15)
            plt.show()
        _, qubit_freq_12, _, _ = excited_sweep_fit_params
        self._qubit_freq_excited = qubit_freq_12
        if verbose:
            print(f"The calibrated frequency for the 1->2 transition\
                     is {self._qubit_freq_excited}.")
        return self._qubit_freq_excited

    def find_pi_amp_excited(self, verbose=False, visual=False):
        """Sets and returns the amplitude of the pi pulse corresponding to 1->2 transition."""
        if not self._qubit_freq_excited:
            warn("ground_qubit_freq not computed yet and custom qubit freq not provided." +
                 "Computing ground_qubit_freq now.")
            self._qubit_freq_ground = self.find_freq_excited(verbose, visual)
        amp_list = np.linspace(0, 1.0, 75)
        rabi_sched_list = []
        for amp in amp_list:
            base_pulse = pulse_lib.gaussian(duration=self._pulse_duration,
                                            sigma=self._pulse_sigma, amp=amp)
            rabi_sched_list.append(self.create_cal_circuit_excited(base_pulse,
                                   self._qubit_freq_excited))
        rabi_job = execute(rabi_sched_list, backend=self._backend,
                            meas_level=1, meas_return='avg', shots=1024,
                            schedule_los=[{self._drive_chan: self._qubit_freq_ground}]*75
                            )
        if verbose:
            print("Executing the rabi oscillation job for 1->2 transition.")
            job_monitor(rabi_job)
        rabi_data = self.get_job_data(rabi_job, average=True)
        rabi_data = np.real(self._baseline_remove(rabi_data))
        if visual:
            print("The amplitude-signal plot of rabi oscillation for 1->2 transition.")
            plt.figure()
            plt.scatter(amp_list, rabi_data, color='black')
            plt.xlabel("Drive amp [a.u.]", fontsize=15)
            plt.ylabel("Measured signal [a.u.]", fontsize=15)
            plt.title('Rabi Experiment (1->2)', fontsize=20)
            plt.show()

        (rabi_fit_params,
         rabi_y_fit) = self._fit_function(amp_list,
                                          rabi_data,
                                          lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                          [3e-8, 0, 0.4, np.pi/2])
        drive_period_excited = rabi_fit_params[2]
        pi_amp_excited = (drive_period_excited/2/np.pi) * (np.pi+rabi_fit_params[3])
        self._pi_amp_excited = pi_amp_excited
        if visual:
            print("The amplitude-signal plot of rabi oscillation for 1->2 transition.")
            plt.figure()
            plt.scatter(amp_list, rabi_data, color='black')
            plt.plot(amp_list, rabi_y_fit, color='red')
            # account for phi in computing pi amp

            plt.axvline(self._pi_amp_excited, color='red', linestyle='--')
            plt.axvline(self._pi_amp_excited+drive_period_excited/2, color='red', linestyle='--')
            plt.annotate("", xy=(self._pi_amp_excited+drive_period_excited/2, 0),
                         xytext=(self._pi_amp_excited, 0),
                         arrowprops=dict(arrowstyle="<->", color='red'))
            #plt.annotate("$\\pi$", xy=(self._pi_amp_excited-0.03, 0.1), color='red')

            plt.xlabel("Drive amp [a.u.]", fontsize=15)
            plt.ylabel("Measured signal [a.u.]", fontsize=15)
            plt.title('Rabi Experiment (1->2)', fontsize=20)
            plt.show()
        return self._pi_amp_excited

    def get_pi_pulse_ground(self):
        """Returns a pi pulse of the 0->1 transition."""
        pulse = pulse_lib.gaussian(duration=self._pulse_duration,
                                   sigma=self._pulse_sigma, amp=self._pi_amp_ground)
        return pulse

    def get_pi_pulse_excited(self):
        """Returns a pi pulse of the 1->2 transition."""
        pulse = pulse_lib.gaussian(duration=self._pulse_duration, sigma=self._pulse_sigma,
                                   amp=self._pi_amp_excited)
        excited_pulse = self.apply_sideband(pulse, self._qubit_freq_excited)
        return excited_pulse

    def get_x90_pulse_ground(self):
        """Returns a pi/2 pulse of the 0->1 transition."""
        pulse = pulse_lib.gaussian(duration=self._pulse_duration,
                                   sigma=self._pulse_sigma, amp=self._pi_amp_ground/2)
        return pulse
    def get_x90_pulse_excited(self):
        """Returns a pi/2 pulse of the 1->2 transition."""
        pulse = pulse_lib.gaussian(duration=self._pulse_duration, sigma=self._pulse_sigma,
                                   amp=self._pi_amp_excited/2)
        excited_pulse = self.apply_sideband(pulse, self._qubit_freq_excited)
        return excited_pulse

    def get_zero_sched(self):
        """Returns a schedule that performs only a measurement."""
        zero_sched = Schedule()
        zero_sched += self._measure
        return zero_sched

    def get_one_sched(self):
        """Returns a schedule that creates a |1> state from |0> by applying
           a pi pulse of 0->1 transition and performs a measurement."""
        one_sched = Schedule()
        one_sched += Play(self.get_pi_pulse_ground(), self._drive_chan)
        one_sched += self._measure << one_sched.duration
        return one_sched

    def get_two_sched(self):
        """Returns a schedule that creates a |2> state from |0> by applying
           a pi pulse of 0->1 transition followed by applying a pi pulse
           of 1->2 transition and performs a measurement."""
        two_sched = Schedule()
        two_sched += Play(self.get_pi_pulse_ground(), self._drive_chan)
        two_sched += Play(self.get_pi_pulse_excited(), self._drive_chan)
        two_sched += self._measure << two_sched.duration
        return two_sched

    @staticmethod
    def _create_iq_plot(zero_data, one_data, two_data):
        """Helper function for plotting IQ plane for 0, 1, 2. Limits of plot given
        as arguments."""
        # zero data plotted in blue
        plt.scatter(np.real(zero_data), np.imag(zero_data),
                    s=5, cmap='viridis', c='blue', alpha=0.5, label=r'$|0\rangle$')
        # one data plotted in red
        plt.scatter(np.real(one_data), np.imag(one_data),
                    s=5, cmap='viridis', c='red', alpha=0.5, label=r'$|1\rangle$')
        # two data plotted in green
        plt.scatter(np.real(two_data), np.imag(two_data),
                    s=5, cmap='viridis', c='green', alpha=0.5, label=r'$|2\rangle$')

        x_min = np.min(np.append(np.real(zero_data), np.append(np.real(one_data),
                       np.real(two_data))))-5
        x_max = np.max(np.append(np.real(zero_data), np.append(np.real(one_data),
                       np.real(two_data))))+5
        y_min = np.min(np.append(np.imag(zero_data), np.append(np.imag(one_data),
                       np.imag(two_data))))-5
        y_max = np.max(np.append(np.imag(zero_data), np.append(np.imag(one_data),
                       np.imag(two_data))))+5

        # Plot a large dot for the average result of the 0, 1 and 2 states.
        mean_zero = np.mean(zero_data)  # takes mean of both real and imaginary parts
        mean_one = np.mean(one_data)
        mean_two = np.mean(two_data)
        plt.scatter(np.real(mean_zero), np.imag(mean_zero),
                    s=200, cmap='viridis', c='black', alpha=1.0)
        plt.scatter(np.real(mean_one), np.imag(mean_one),
                    s=200, cmap='viridis', c='black', alpha=1.0)
        plt.scatter(np.real(mean_two), np.imag(mean_two),
                    s=200, cmap='viridis', c='black', alpha=1.0)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.ylabel('I [a.u.]', fontsize=15)
        plt.xlabel('Q [a.u.]', fontsize=15)
        plt.title("0-1-2 discrimination", fontsize=15)
        return x_min, x_max, y_min, y_max

    @staticmethod
    def reshape_complex_vec(vec):
        """Take in complex vector vec and return 2d array w/ real, imag entries.
           This is needed for the learning.
        Args:
            vec (list): complex vector of data
        Returns:
            list: vector w/ entries given by (real(vec], imag(vec))
        """
        length = len(vec)
        vec_reshaped = np.zeros((length, 2))
        for i, item in enumerate(vec):
            vec_reshaped[i] = [np.real(item), np.imag(item)]
        return vec_reshaped

    def find_three_level_discriminator(self, verbose=False, visual=False):
        """Returns a discriminator for discriminating 0-1-2 states."""
        # pylint: disable=too-many-locals
        zero_sched = self.get_zero_sched()
        one_sched = self.get_one_sched()
        two_sched = self.get_two_sched()
        iq_job = execute([zero_sched, one_sched, two_sched], backend=self._backend,
                          meas_level=1, meas_return='single', shots=1024,
                          schedule_los=[{self._drive_chan: self._qubit_freq_ground}]*3
                          )
        if verbose:
            job_monitor(iq_job)
        iq_data = self.get_job_data(iq_job, average=False)
        zero_data = iq_data[0]
        one_data = iq_data[1]
        two_data = iq_data[2]
        if visual:
            print("The discriminator plot for 0-1-2 discrimination.")
            x_min, x_max, y_min, y_max = self._create_iq_plot(zero_data, one_data, two_data)
        # Create IQ vector (split real, imag parts)
        zero_data_reshaped = self.reshape_complex_vec(zero_data)
        one_data_reshaped = self.reshape_complex_vec(one_data)
        two_data_reshaped = self.reshape_complex_vec(two_data)

        iq_data = np.concatenate((zero_data_reshaped, one_data_reshaped, two_data_reshaped))

        # construct vector w/ 0's, 1's and 2's (for testing)
        state_012 = np.zeros(1024)  # shots gives number of experiments
        state_012 = np.concatenate((state_012, np.ones(1024)))
        state_012 = np.concatenate((state_012, 2*np.ones(1024)))

        # Shuffle and split data into training and test sets
        iq_012_train, iq_012_test, state_012_train, state_012_test = train_test_split(iq_data,
                                                                                      state_012,
                                                                                      test_size=0.5)
        classifier_lda_012 = LinearDiscriminantAnalysis()
        classifier_lda_012.fit(iq_012_train, state_012_train)
        score_012 = classifier_lda_012.score(iq_012_test, state_012_test)
        if verbose:
            print('The accuracy score of the discriminator is: ', score_012)
        self._state_discriminator_012 = classifier_lda_012
        if visual:
            self._separatrix_plot(classifier_lda_012, x_min, x_max, y_min, y_max, 1024)
        return self._state_discriminator_012

    @staticmethod
    def _separatrix_plot(lda, x_min, x_max, y_min, y_max, shots):
        """Returns a sepratrix plot for the classifier."""
        # pylint: disable=too-many-arguments
        num_x, num_y = shots, shots
        xvals, vals = np.meshgrid(np.linspace(x_min, x_max, num_x),
                                  np.linspace(y_min, y_max, num_y))
        predict_prob = lda.predict_proba(np.c_[xvals.ravel(), vals.ravel()])
        predict_prob = predict_prob[:, 1].reshape(xvals.shape)
        plt.contour(xvals, vals, predict_prob, [0.5], linewidths=2., colors='black')

    def get_qubit_freq_ground(self):
        """Returns the set 0->1 transition frequency."""
        return self._qubit_freq_ground

    def get_qubit_freq_excited(self):
        """Returns the set 1->2 transition frequency."""
        return self._qubit_freq_excited

    def get_pi_amp_ground(self):
        """Returns the set 0->1 transition pi pulse amplitude."""
        return self._pi_amp_ground

    def get_pi_amp_excited(self):
        """Returns the set 1->2 transition pi pulse amplitude."""
        return self._pi_amp_excited

    def get_three_level_discriminator(self):
        """Returns the set 0-1-2 state discriminator."""
        return self._state_discriminator_012

    def calibrate_all(self, verbose=False, visual=False):
        """Calibrates and sets both the ground and excited transition frequencies and
           corresponding pi pulse amplitudes. Also constructs a 0-1-2 state discriminator."""
        ground_freq = self.find_freq_ground(verbose, visual)
        ground_amp = self.find_pi_amp_ground(verbose, visual)
        excited_freq = self.find_freq_excited(verbose, visual)
        excited_amp = self.find_pi_amp_excited(verbose, visual)
        state_discriminator = self.find_three_level_discriminator(verbose, visual)
        return ground_freq, ground_amp, excited_freq, excited_amp, state_discriminator
    def T1_ground(self, verbose=False, visual=False):
        """Returns the T1 of 1 state"""
        time_max=600
        time_step=8
        times=np.arange(1,time_max,time_step)*1e-6
        
        delay_times_dt=times/self._dt

        t1_gnd_sched=[]
        for delay in delay_times_dt:
            this_sched= Schedule(name=f"T1 delay={delay*self._dt/1e-6} us")
            this_sched+= Play(self.get_pi_pulse_ground(),self._drive_chan)
            this_sched|=self._measure<<int(delay)
            t1_gnd_sched.append(this_sched)
        #sched_idx=0
        #t1_gnd_sched[sched_idx]
        # Execution settings
        t1_gnd_job = execute(t1_gnd_sched, backend=self._backend, meas_level=1,
                             meas_return='avg', shots=256,
                             schedule_los=[{self._drive_chan: self.get_qubit_freq_ground()}] 
                             * len(t1_gnd_sched))
        if verbose:
            job_monitor(t1_gnd_job)
        t1_gnd_results =self.get_job_data(t1_gnd_job, average=True)
        t1_gnd_results = np.real(t1_gnd_results)
        # Fit the data
        fit_params, y_fit = self._fit_function(times, t1_gnd_results, 
        lambda x, A, C, T1: (A * np.exp(-x / T1) + C),[-6, 8, 100*1e-6])

        _, _, T1_gnd = fit_params
        if visual:
            plt.scatter(times*1e6, t1_gnd_results, color='black')
            plt.plot(times*1e6, y_fit, color='red', label=f"T1 = {T1_gnd*1e6:.2f} us")
            plt.xlim(0, np.max(times)*1e6)
            plt.title("$T_1$ Experiment", fontsize=15)
            plt.xlabel('Delay before measurement [$\mu$s]', fontsize=15)
            plt.ylabel('Signal [a.u.]', fontsize=15)
            plt.legend()
            plt.show()
        return T1_gnd
    def T1_excited(self, verbose=False, visual=False):
        """Returns the T1 of 2 state"""
        time_max=600
        time_step=8
        times=np.arange(1,time_max,time_step)*1e-6
        
        delay_times_dt=times/self._dt

        t1_exc_sched=[]
        for delay in delay_times_dt:
            this_sched= Schedule(name=f"T1 delay={delay*self._dt/1e-6} us")
            this_sched+= Play(self.get_pi_pulse_ground(),self._drive_chan)
            this_sched += Play(self.get_pi_pulse_excited(), self._drive_chan)
            this_sched|=self._measure<<int(delay)
            t1_exc_sched.append(this_sched)
        #sched_idx=0
        #t1_gnd_sched[sched_idx]
        # Execution settings
        t1_exc_job = execute(t1_exc_sched, backend=self._backend, meas_level=1,
                             meas_return='avg', shots=256,
                             schedule_los=[{self._drive_chan: self.get_qubit_freq_ground()}] 
                             * len(t1_exc_sched))
        if verbose:
            job_monitor(t1_exc_job)
        t1_exc_results =self.get_job_data(t1_exc_job, average=True)
        t1_exc_results = np.real(t1_exc_results) # Reverse the sign to have a decay behavior.
        # Fit the data
        fit_params, y_fit = self._fit_function(times, t1_exc_results, lambda x, A, C, T1: (A * np.exp(-x / T1) + C),[-6, 8, 100*1e-6])

        _, _, T1_exc = fit_params
        if visual:
            plt.scatter(times*1e6, t1_exc_results, color='black')
            plt.plot(times*1e6, y_fit, color='red', label=f"T1 = {T1_exc*1e6:.2f} us")
            plt.xlim(0, np.max(times)*1e6)
            plt.title("$T_1$ Experiment", fontsize=15)
            plt.xlabel('Delay before measurement [$\mu$s]', fontsize=15)
            plt.ylabel('Signal [a.u.]', fontsize=15)
            plt.legend()
            plt.show()
        return T1_exc
    def T2_ground(self,verbose=False,visual=False):
        """Returns the T2 of 1 state"""
        time_max=300
        time_step=4
        times=np.arange(2,time_max,time_step)*1e-6
        
        delay_times_dt=times/self._dt

        t2_gnd_sched=[]
        for delay in delay_times_dt:
            this_sched= Schedule(name=f"T2 delay={delay*self._dt/1e-6} us")
            this_sched|= Play(self.get_x90_pulse_ground(),self._drive_chan)
            this_sched|= Play(self.get_pi_pulse_ground(),self._drive_chan) << int(this_sched.duration+delay)
            this_sched|= Play(self.get_x90_pulse_ground(),self._drive_chan)<< int(this_sched.duration+delay)
            this_sched|=self._measure<<int(this_sched.duration)
            t2_gnd_sched.append(this_sched)
        #sched_idx=0
        #t1_gnd_sched[sched_idx]
        # Execution settings
        t2_gnd_job = execute(t2_gnd_sched, backend=self._backend, meas_level=1,
                             meas_return='avg', shots=512,
                             schedule_los=[{self._drive_chan: self.get_qubit_freq_ground()}] 
                             * len(t2_gnd_sched))
        if verbose:
            job_monitor(t2_gnd_job)
        t2_gnd_results =self.get_job_data(t2_gnd_job, average=True)
        t2_gnd_results = np.real(t2_gnd_results)
        # Fit the data2
        fit_params, y_fit = self._fit_function(2*times, t2_gnd_results, lambda x, A, C, T2: (A * np.exp(-x / T2) + C),[-3, 0, 100*1e-6])

        _, _, T2_gnd = fit_params
        if visual:
            plt.scatter(2*times*1e6, t2_gnd_results, color='black')
            plt.plot(2*times*1e6, y_fit, color='red', label=f"T2 = {T2_gnd*1e6:.2f} us")
            plt.xlim(0, np.max(2*times)*1e6)
            plt.title("$T_2$ Experiment with Hahn Echo", fontsize=15)
            plt.xlabel('Delay before measurement [$\mu$s]', fontsize=15)
            plt.ylabel('Signal [a.u.]', fontsize=15)
            plt.legend()
            plt.show()
        return T2_gnd