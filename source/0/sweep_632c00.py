# https://github.com/panbae59/smooth-optimal-control-2/blob/8fe55d599a70e03fe27ead12e0b25a1a6c1ac099/util/sweep.py
import numpy as np
from util.utils import plot_shot_results
from qiskit import pulse
from qiskit.pulse import Play
from qiskit.pulse import library as pulse_lib
from qiskit.visualization.pulse_v2 import draw
from qiskit import assemble
from qiskit.tools.monitor import job_monitor
from qiskit.pulse import ShiftPhase

class Sweep:
    def __init__(self, Q_setup):
        self.Q_setup = Q_setup
    def amplitude_sweep(self, drive_amps : np.array, Signal_list):
        # setup schedules
        schedules = []
        for drive_amp in drive_amps:
            this_schedule = pulse.Schedule(name=f"amplitude = {drive_amp}")
            for Signal in Signal_list:
                this_schedule += Play(pulse_lib.Waveform(Signal*drive_amp), self.Q_setup.drive_chan)
            this_schedule += self.Q_setup.measure << this_schedule.duration
            schedules.append(this_schedule)
            
        draw(schedules[-1], backend = self.Q_setup.backend)
        
        # generate experiment program
        num_shots_per_point = 1024
        experiment_program = assemble(schedules,
                                    backend=self.Q_setup.backend,
                                    meas_level=2,
                                    meas_return='avg',
                                    shots=num_shots_per_point,
                                    schedule_los=[{self.Q_setup.drive_chan: self.Q_setup.center_frequency_Hz}]
                                                    * len(schedules))
        
        # run program and get results
        job = self.Q_setup.backend.run(experiment_program)
        job_monitor(job)
        results = job.result(timeout=120)
        plot_shot_results(job)
        return results.get_counts()

    def angle_sweep_pi(self, angles, Signal_list, base_gate_X_over_2):
        schedules = []
        for angle in angles:
            this_schedule = pulse.Schedule(name=f"rotating angle = {angle}")
            this_schedule += base_gate_X_over_2
            for Signal in Signal_list:
                this_schedule += ShiftPhase(angle, self.Q_setup.drive_chan)
                this_schedule += Play(pulse_lib.Waveform(Signal), self.Q_setup.drive_chan)
                this_schedule += ShiftPhase(-angle, self.Q_setup.drive_chan)
            # Reuse the measure instruction from the frequency sweep experiment
            this_schedule += base_gate_X_over_2
            this_schedule += self.Q_setup.measure << this_schedule.duration
            schedules.append(this_schedule)
            
        draw(schedules[-1], backend = self.Q_setup.backend) # is not working! (could be a problem of jupyter)
        # Assemble the schedules into a Qobj
        num_shots_per_point = 1024

        experiment_program = assemble(schedules,
                                    backend=self.Q_setup.backend,
                                    meas_level=2,
                                    meas_return='avg',
                                    shots=num_shots_per_point,
                                    schedule_los=[{self.Q_setup.drive_chan: self.Q_setup.center_frequency_Hz}]
                                                    * len(schedules))
        job = self.Q_setup.backend.run(experiment_program)
        job_monitor(job)
        results = job.result(timeout=120)
        plot_shot_results(job)
        return results.get_counts()

    def angle_sweep_pi_over_2(self, angles, Signal_list, base_pi_over_2):
        schedules = []
        for angle in angles:
            this_schedule = pulse.Schedule(name=f"rotating angle = {angle}")
            this_schedule += base_pi_over_2
            for Signal in Signal_list:
                this_schedule += ShiftPhase(angle, self.Q_setup.drive_chan)
                this_schedule += Play(pulse_lib.Waveform(Signal), self.Q_setup.drive_chan)
                this_schedule += ShiftPhase(-angle, self.Q_setup.drive_chan)
            # Reuse the measure instruction from the frequency sweep experiment
            this_schedule += self.Q_setup.measure << this_schedule.duration
            schedules.append(this_schedule)
            
        draw(schedules[-1], backend = self.Q_setup.backend) # is not working! (could be a problem of jupyter)
        
        # Assemble the schedules into a Qobj
        num_shots_per_point = 1024

        experiment_program = assemble(schedules,
                                    backend=self.Q_setup.backend,
                                    meas_level=2,
                                    meas_return='avg',
                                    shots=num_shots_per_point,
                                    schedule_los=[{self.Q_setup.drive_chan: self.Q_setup.center_frequency_Hz}]
                                                    * len(schedules))
        job = self.Q_setup.backend.run(experiment_program)
        job_monitor(job)
        results = job.result(timeout=120)
        plot_shot_results(job)
        return results.get_counts()

    def frequency_sweep(self, pulse_list : list):
        frequency_span_Hz = 4 * self.Q_setup.MHz
        frequency_step_Hz = 0.1 * self.Q_setup.MHz

        # We will sweep 20 MHz above and 20 MHz below the estimated frequency
        frequency_min = self.Q_setup.center_frequency_Hz - frequency_span_Hz / 2
        frequency_max = self.Q_setup.center_frequency_Hz + frequency_span_Hz / 2
        # Construct an np array of the frequencies for our experiment
        frequencies_GHz = np.arange(frequency_min / self.Q_setup.GHz, 
                                    frequency_max / self.Q_setup.GHz, 
                                    frequency_step_Hz / self.Q_setup.GHz)

        print(f"The sweep will go from {frequency_min / self.Q_setup.GHz} GHz to {frequency_max / self.Q_setup.GHz} GHz \
        in steps of {frequency_step_Hz / self.Q_setup.MHz} MHz.")
        
        
        # Create the base schedule
        # Start with drive pulse acting on the drive channel
        schedule = pulse.Schedule(name='Frequency sweep')
        for p in pulse_list:
            schedule += p
        # The left shift `<<` is special syntax meaning to shift the start time of the schedule by some duration
        schedule += self.Q_setup.measure << schedule.duration

        # Create the frequency settings for the sweep (MUST BE IN HZ)
        frequencies_Hz = frequencies_GHz*self.Q_setup.GHz
        schedule_frequencies = [{self.Q_setup.drive_chan: freq} for freq in frequencies_Hz]
        
        num_shots_per_frequency = 1024
        pi_frequency_sweep_program = assemble(schedule,
                                        backend=self.Q_setup.backend, 
                                        meas_level=2,
                                        meas_return='avg',
                                        shots=num_shots_per_frequency,
                                        schedule_los=schedule_frequencies)
        job = self.Q_setup.backend.run(pi_frequency_sweep_program)
        job_monitor(job)
        plot_shot_results(job)