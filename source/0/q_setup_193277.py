# https://github.com/snuphya/smooth-optimal-control-2/blob/0741829b3350b8134b5a7f14951e806bc50c7bc8/util/Q_setup.py
from qiskit import IBMQ
from qiskit import pulse

class QSetUp:
    def __init__(self,):
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        self.backend = provider.get_backend('ibmq_armonk')
        backend_config = self.backend.configuration()
        assert backend_config.open_pulse, "Backend doesn't support Pulse"

        self.dt = backend_config.dt
        print(f"Sampling time: {self.dt*1e9} ns")
        backend_defaults = self.backend.defaults()

        import numpy as np

        # unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
        self.GHz = 1.0e9 # Gigahertz
        self.MHz = 1.0e6 # Megahertz
        self.us = 1.0e-6 # Microseconds
        self.ns = 1.0e-9 # Nanoseconds

        # We will find the qubit frequency for the following qubit.
        qubit = 0

        # The sweep will be centered around the estimated qubit frequency.
        self.center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]        # The default frequency is given in Hz
                                                                            # warning: this will change in a future release
        print(f"Qubit {qubit} has an estimated frequency of {self.center_frequency_Hz / self.GHz} GHz.")

        # Find out which group of qubits need to be acquired with this qubit
        meas_map_idx = None
        for i, measure_group in enumerate(backend_config.meas_map):
            if qubit in measure_group:
                meas_map_idx = i
                break
        assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"

        ### Collect the necessary channels
        self.drive_chan = pulse.DriveChannel(qubit)
        self.meas_chan = pulse.MeasureChannel(qubit)
        self.acq_chan = pulse.AcquireChannel(qubit)

        inst_sched_map = backend_defaults.instruction_schedule_map
        self.measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])