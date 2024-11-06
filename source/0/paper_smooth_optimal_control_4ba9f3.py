# https://github.com/panbae59/smooth-optimal-control-2/blob/8fe55d599a70e03fe27ead12e0b25a1a6c1ac099/optimal_control/paper_smooth_optimal_control.py
import numpy as np
from numpy.lib.npyio import save
from numpy.lib.shape_base import get_array_wrap
from util.utils import get_closest_multiple_of_16
from util.utils import baseline_remove, fit_function
from optimal_control.optimal_control import OptimalControl
from util.utils import save_value, load_value, get_closest_multiple_of_16, load_Signal
from qiskit.pulse import library as pulse_lib
from qiskit.pulse import Play
from qiskit.pulse import ShiftPhase
from qiskit import pulse
from util.sweep import Sweep

class PaperSmoothOptimalControl(OptimalControl):
    def __init__(self, Q_setup):
        super().__init__()
        self.Q_setup = Q_setup
        self.sweep = Sweep(self.Q_setup)
        self.set_pulse_params()
        self.set_gates()

    def set_pulse_params(self, ):
        # setup for pi Signal
        self.drive_samples_pi = int(500e-9/self.Q_setup.dt)
        self.omega_pi = np.pi/(500*1e-9)
        self.drive_samples_pi = get_closest_multiple_of_16(self.drive_samples_pi)

        # setup for pi_over_2 Signal
        self.drive_samples_pi_over_2 = int(250e-9/self.Q_setup.dt)
        self.omega_pi_over_2 = np.pi/(250*1e-9)
        self.drive_samples_pi_over_2 = get_closest_multiple_of_16(self.drive_samples_pi_over_2)

    def get_raw_SOC_pi_Signal(self):
        Signal = load_Signal(self.omega_pi, self.Q_setup.dt, self.drive_samples_pi, "amplitudes/paper/amplitude_pi_X_x.csv", "amplitudes/paper/amplitude_pi_X_y.csv")
        return Signal

    def get_raw_SOC_pi_over_2_Signal(self):
        Signal = load_Signal(self.omega_pi_over_2, self.Q_setup.dt, self.drive_samples_pi_over_2, "amplitudes/paper/amplitude_pi_over_2_X_x.csv", "amplitudes/paper/amplitude_pi_over_2_X_y.csv")
        return Signal

    def set_gates(self, ):
        #load value...
        Signal = self.get_raw_SOC_pi_Signal()
        amplitude_multiplier_paper_SOC = load_value('amplitude_multiplier_paper_SOC')
        paper_SOC_pi_angle = load_value('paper_SOC_pi_angle')

        Signal_final = Signal * amplitude_multiplier_paper_SOC
        pulse_paper_SOC_pi_X = pulse.Waveform(Signal_final)
        pulse_paper_SOC_pi_Y = pulse.Waveform(Signal_final*1j)

        self.gate['X'] = ShiftPhase(paper_SOC_pi_angle, self.Q_setup.drive_chan) + Play(pulse_paper_SOC_pi_X, self.Q_setup.drive_chan) + ShiftPhase(-paper_SOC_pi_angle, self.Q_setup.drive_chan)
        self.gate['Y'] = ShiftPhase(paper_SOC_pi_angle, self.Q_setup.drive_chan) + Play(pulse_paper_SOC_pi_Y, self.Q_setup.drive_chan) + ShiftPhase(-paper_SOC_pi_angle, self.Q_setup.drive_chan)

        Signal = self.get_raw_SOC_pi_over_2_Signal()
        amplitude_multiplier_paper_SOC_pi_over_2 = load_value('amplitude_multiplier_paper_SOC_pi_over_2')
        paper_SOC_pi_over_2_angle = load_value('paper_SOC_pi_over_2_angle')
        Signal_final = Signal * amplitude_multiplier_paper_SOC_pi_over_2
        pulse_paper_SOC_pi_X_over_2 = pulse.Waveform(Signal_final)
        pulse_paper_SOC_pi_Y_over_2 = pulse.Waveform(Signal_final*1j)
        pulse_paper_SOC_pi_m_X_over_2 = pulse.Waveform(-Signal_final)
        pulse_paper_SOC_pi_m_Y_over_2 = pulse.Waveform(-Signal_final*1j)

        self.gate['X/2'] = ShiftPhase(paper_SOC_pi_over_2_angle, self.Q_setup.drive_chan) + Play(pulse_paper_SOC_pi_X_over_2, self.Q_setup.drive_chan) + ShiftPhase(-paper_SOC_pi_over_2_angle, self.Q_setup.drive_chan)
        self.gate['Y/2'] = ShiftPhase(paper_SOC_pi_over_2_angle, self.Q_setup.drive_chan) + Play(pulse_paper_SOC_pi_Y_over_2, self.Q_setup.drive_chan) + ShiftPhase(-paper_SOC_pi_over_2_angle, self.Q_setup.drive_chan)
        self.gate['-X/2'] = ShiftPhase(paper_SOC_pi_over_2_angle, self.Q_setup.drive_chan) + Play(pulse_paper_SOC_pi_m_X_over_2, self.Q_setup.drive_chan) + ShiftPhase(-paper_SOC_pi_over_2_angle, self.Q_setup.drive_chan)
        self.gate['-Y/2'] = ShiftPhase(paper_SOC_pi_over_2_angle, self.Q_setup.drive_chan) + Play(pulse_paper_SOC_pi_m_Y_over_2, self.Q_setup.drive_chan) + ShiftPhase(-paper_SOC_pi_over_2_angle, self.Q_setup.drive_chan)
    
    def tune_paper_SOC_pulse(self, base_gate_X_over_2):
        self.tune_amplitude_pi()
        self.tune_angle_pi(base_gate_X_over_2)
        self.tune_amplitude_pi_over_2()
        self.tune_angle_pi_over_2(base_gate_X_over_2)

    def tune_amplitude_pi(self, ):
        Signal = self.get_raw_SOC_pi_Signal()

        amplitude_detuning_array = np.linspace(0.08,0.12, 50)
        measure_list = self.sweep.amplitude_sweep(amplitude_detuning_array, [Signal])
        fit_params, y_fit = fit_function(amplitude_detuning_array[:],
                                 [x['0'] for x in measure_list][:], 
                                 lambda x, A, B, C: (A*(x-B)**2 + C),
                                 [100000, 1, 0])

        from matplotlib import pyplot as plt
        plt.scatter(amplitude_detuning_array[:], [x['0'] for x in measure_list][:])
        plt.plot(amplitude_detuning_array[:], y_fit)
        plt.show()

        amplitude_multiplier_paper_SOC = fit_params[1]
        print("amplitude_multiplier = {}".format(amplitude_multiplier_paper_SOC))
        save_value("amplitude_multiplier_paper_SOC", amplitude_multiplier_paper_SOC)


    def tune_angle_pi(self, base_gate_X_over_2):
        amplitude_multiplier_paper_SOC = load_value("amplitude_multiplier_paper_SOC")
        angles = np.linspace(0, 2*np.pi, 50)
        Signal = self.get_raw_SOC_pi_Signal()
        measure_list = self.sweep.angle_sweep_pi(angles, [Signal*amplitude_multiplier_paper_SOC], base_gate_X_over_2)
        
        values = baseline_remove([x['0'] for x in measure_list])
        fit_params, y_fit = fit_function(angles,
                                        values, 
                                        lambda x, A, B, phi: (A*np.cos(2*(x - phi)) + B),
                                        [490, 0.8, 0])

        from matplotlib import pyplot as plt

        plt.scatter(angles, values, color='black')
        plt.plot(angles, y_fit, color='red')

        plt.xlabel("Drive amp [a.u.]", fontsize=15)
        plt.ylabel("Measured signal [a.u.]", fontsize=15)
        plt.show()

        paper_SOC_pi_angle = fit_params[2] if fit_params[0] > 0 else fit_params[2] + np.pi/2

        save_value("paper_SOC_pi_angle", paper_SOC_pi_angle)

    def tune_amplitude_pi_over_2(self):
        Signal = self.get_raw_SOC_pi_over_2_Signal()
        amplitude_detuning_array = np.linspace(0.025,0.04, 20)
        measure_list = self.sweep.amplitude_sweep(amplitude_detuning_array, [Signal, Signal])
        
        fit_params, y_fit = fit_function(amplitude_detuning_array[:],
                                 [x['0'] for x in measure_list][:], 
                                 lambda x, A, B, C: (A*(x-B)**2 + C),
                                 [100000, 1, 0])

        from matplotlib import pyplot as plt
        plt.scatter(amplitude_detuning_array[:], [x['0'] for x in measure_list][:])
        plt.plot(amplitude_detuning_array[:], y_fit)
        plt.show()

        amplitude_multiplier_paper_SOC_pi_over_2 = fit_params[1]
        print("amplitude_multiplier = {}".format(amplitude_multiplier_paper_SOC_pi_over_2))
        save_value("amplitude_multiplier_paper_SOC_pi_over_2", amplitude_multiplier_paper_SOC_pi_over_2)

    def tune_angle_pi_over_2(self, base_gate_X_over_2):
        Signal = self.get_raw_SOC_pi_over_2_Signal()
        amplitude_multiplier_paper_SOC_pi_over_2 = load_value("amplitude_multiplier_paper_SOC_pi_over_2")
        angles = np.linspace(0, 2*np.pi, 50)
        measure_list = self.sweep.angle_sweep_pi_over_2(angles, [Signal * amplitude_multiplier_paper_SOC_pi_over_2], base_gate_X_over_2)

        values = baseline_remove([x['1'] for x in measure_list])
        fit_params, y_fit = fit_function(angles,
                                        values, 
                                        lambda x, A, B, phi: (A*np.cos(x - phi) + B),
                                        [490, 0.8, 0])

        from matplotlib import pyplot as plt
        plt.scatter(angles, values, color='black')
        plt.plot(angles, y_fit, color='red')

        plt.xlabel("Drive amp [a.u.]", fontsize=15)
        plt.ylabel("Measured signal [a.u.]", fontsize=15)
        plt.show()

        paper_SOC_pi_over_2_angle = fit_params[2]
        save_value("paper_SOC_pi_over_2_angle", paper_SOC_pi_over_2_angle)
