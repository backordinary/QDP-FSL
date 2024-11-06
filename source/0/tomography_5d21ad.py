# https://github.com/panbae59/smooth-optimal-control-2/blob/0741829b3350b8134b5a7f14951e806bc50c7bc8/util/tomography.py
import numpy as np
from qiskit import pulse
from qiskit.visualization.pulse_v2 import draw
from qiskit import assemble
from qiskit.tools.monitor import job_monitor
from scipy.optimize import dual_annealing
import pandas as pd
import copy
import os

class Tomography:
    def __init__(self, Q_setup):
        self.Q_setup = Q_setup
        self.pauli_x = np.array([[0,1],[1,0]])
        self.pauli_y = np.array([[0,-1j],[1j,0]])
        self.pauli_z = np.array([[1,0],[0,-1]])
        self.basis_matrices = np.array([np.identity(2), self.pauli_x, -1j*self.pauli_y, self.pauli_z])
        initial_density_matrices = np.array([np.array([[1,0]]).transpose()@np.array([[1,0]]).conjugate(),\
                                            np.array([[0,1]]).transpose()@np.array([[0,1]]).conjugate(),\
                                            1/2*np.array([[1,1]]).transpose()@np.array([[1,1]]).conjugate(),\
                                            1/2*np.array([[1,1j]]).transpose()@np.array([[1,1j]]).conjugate()])

        self.inverse_initial_density_matrices = np.linalg.inv(initial_density_matrices.transpose(1,2,0).reshape((4,4),order = 'F')).transpose()

    def calculate_state_fidelity(self, x_expect_val, y_expect_val, z_expect_val, target_state_vector):
        pauli_x = np.array([[0,1],[1,0]])
        pauli_y = np.array([[0,-1j],[1j,0]])
        pauli_z = np.array([[1,0],[0,-1]])
        density_matrix = 1/2 * (np.identity(2) + x_expect_val * pauli_x + y_expect_val * pauli_y + z_expect_val * pauli_z)
        print(density_matrix)
        fidelity = np.dot(target_state_vector.conjugate(), density_matrix@(target_state_vector))
        print(fidelity)
        return fidelity

    def get_expectation_value_with_detuning(self, schedule, detuning, num = 1, execute = True):
        num_shots_per_point = 1024
        new_Freq = self.Q_setup.center_frequency_Hz + detuning
        program = assemble(schedule,backend=self.Q_setup.backend,meas_level=2,meas_return='avg',shots=num_shots_per_point,schedule_los=[{self.Q_setup.drive_chan: new_Freq}]*num)

        if execute:
            job = self.Q_setup.backend.run(program)
            job_monitor(job)
        results = job.result(timeout=120)
        measure_list = results.get_counts()
        print(measure_list)
        if num ==1:
            expectation_value = (measure_list['0'] * 1 + measure_list['1'] * (-1)) / (measure_list['0'] + measure_list['1'])
        else:
            expectation_value = (measure_list[-1]['0'] * 1 + measure_list[-1]['1'] * (-1)) / (measure_list[-1]['0'] + measure_list[-1]['1'])
        return expectation_value
    def density_matrix(self, x_expect_val, y_expect_val, z_expect_val):
        density_matrix = 1/2 * (np.identity(2) + x_expect_val * self.pauli_x + y_expect_val * self.pauli_y + z_expect_val * self.pauli_z)
        print(density_matrix)
        return density_matrix
    def calculate_process_matrix(self, density_matrices):
        id2 = np.identity(2)
        transformed_matrices = (self.inverse_initial_density_matrices@density_matrices.reshape(4,4)).transpose().reshape((4,2,2), order = 'F')

        process_matrix = 1/4 * (np.vstack((np.hstack((id2, self.pauli_x)), np.hstack((self.pauli_x, -id2))))@\
        np.vstack((np.hstack((transformed_matrices[0], transformed_matrices[1])), np.hstack((transformed_matrices[2], transformed_matrices[3]))))@\
        np.vstack((np.hstack((id2, self.pauli_x)), np.hstack((self.pauli_x, -id2)))))
        return process_matrix
    
    def process_matrix_error(self, t,*process_matrix):
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16 = t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]
        T = np.array([[t1, 0, 0, 0],[t5 + 1j*t6, t2, 0, 0],[t11 + 1j*t12, t7+ 1j*t8, t3, 0],[t15 + 1j*t16, t13 + 1j*t14, t9 + 1j*t10, t4]], dtype = np.complex64)
        new_process_matrix = T.conjugate().transpose()@T
        matrix_error = process_matrix - new_process_matrix
        return np.sqrt(np.trace(matrix_error@(matrix_error.conjugate().transpose())))
    
    def adjust_process_matrix(self, process_matrix):
        lw = [-5] * 16
        up = [5] * 16
        ret = dual_annealing(self.process_matrix_error, args = (process_matrix), bounds=list(zip(lw, up)))
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16 = ret.x[0], ret.x[1], ret.x[2], ret.x[3], ret.x[4], ret.x[5], ret.x[6], ret.x[7], ret.x[8], ret.x[9], ret.x[10], ret.x[11], ret.x[12], ret.x[13], ret.x[14], ret.x[15]
        T = np.array([[t1, 0, 0, 0],[t5 + 1j*t6, t2, 0, 0],[t11 + 1j*t12, t7+ 1j*t8, t3, 0],[t15 + 1j*t16, t13 + 1j*t14, t9 + 1j*t10, t4]], dtype = np.complex64)
        new_process_matrix = (T.conjugate().transpose())@T
        return new_process_matrix

    def x_expect_schedule(self, schedule, oc_gate):
        schedule += oc_gate['-Y/2']
        schedule += self.Q_setup.measure << schedule.duration
        return schedule
        #return get_expectation_value_with_detuning(schedule, detuninig)

    def y_expect_schedule(self, schedule, oc_gate):
        schedule += oc_gate['X/2']
        schedule += self.Q_setup.measure << schedule.duration
        return schedule
        # -get_expectation_value_with_detuning(schedule, detuninig)

    def z_expect_schedule(self, schedule, oc_gate):
        schedule += self.Q_setup.measure << schedule.duration
        return schedule
        # return get_expectation_value_with_detuning(schedule, detuninig)
    
    def initialized_schedule(self, init_state, oc_gate):
        schedule = pulse.Schedule()
        if init_state == '0':
            pass
        elif init_state == '1':
            schedule += oc_gate['X']
        elif init_state == '+':
            schedule += oc_gate['Y/2']
        elif init_state == '-':
            schedule += oc_gate['-X/2']
        else:
            assert "init_state is not stated in this function"
        return schedule
    # objective_gate is an obective gate to calculate gate fidelity
    # oc_gate is a category(smooth_optimal_control or gaussian_optimal_control ...) of base gates(ex X, Y, X/2, etc.)
    
    def get_xyz_schedules(self, objective_gate, oc_gate, init_state):
        schedule = pulse.Schedule()
        schedule_init = self.initialized_schedule(init_state, oc_gate)
        schedule_init += objective_gate
        x_schedule = self.x_expect_schedule(copy.deepcopy(schedule_init), oc_gate)
        z_schedule = self.z_expect_schedule(copy.deepcopy(schedule_init), oc_gate)
        y_schedule = self.y_expect_schedule(copy.deepcopy(schedule_init), oc_gate)
        return [x_schedule, y_schedule, z_schedule]
        
    def get_density_matrices_from_schedules(self, pi_xyz_schedules, detuning):
        schedules = np.array(pi_xyz_schedules).flatten().tolist()
        
        draw(schedules[-1], backend = self.Q_setup.backend) # is not working! (could be a problem of jupyter)
        
        # Assemble the schedules into a Qobj
        num_shots_per_point = 1024

        experiment_program = assemble(schedules,
                                    backend=self.Q_setup.backend,
                                    meas_level=2,
                                    meas_return='avg',
                                    shots=num_shots_per_point,
                                    schedule_los=[{self.Q_setup.drive_chan: self.Q_setup.center_frequency_Hz + detuning}]
                                                    * len(schedules))
        job = self.Q_setup.backend.run(experiment_program)
        job_monitor(job)
        results = job.result(timeout=120)
        measure_list = results.get_counts()
        expectation_values_list = [(M['0'] * 1 + M['1'] * (-1)) / (M['0'] + M['1']) for M in measure_list]
        expectation_values_array = np.array(expectation_values_list).reshape(4, 3)
        density_matrices = [self.density_matrix(expectation[0], expectation[1], expectation[2]) for expectation in expectation_values_array]
        return np.array(density_matrices)

    def gate_tomography(self, objective_gate, oc_gate, detuning):
        init_states = ['0', '1', '+', '-']
        xyz_schedules = [self.get_xyz_schedules(objective_gate, oc_gate, init_state) for init_state in init_states]
        pi_density_matrices = self.get_density_matrices_from_schedules(xyz_schedules, detuning)
        pi_process_matrix = self.calculate_process_matrix(pi_density_matrices)
        self.adjust_process_matrix(pi_process_matrix)
        return pi_process_matrix
    
    def plot_density_matrix(self, data_array):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_data, y_data = np.meshgrid( np.arange(4),
                                    np.arange(4))
        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = data_array.flatten()
        ax.bar3d( x_data,
                y_data,
                0.5*np.zeros(len(z_data)),
                0.8, 0.8, z_data )
                
        plt.show()
    
    def gate_fidelity(self, result_process_matrix : np.array, ideal_process_matrix : np.array):
        return np.trace(np.matrix(result_process_matrix) @ np.array(np.matrix(ideal_process_matrix).getH())) \
            /np.sqrt(np.trace(np.matrix(result_process_matrix) @ np.array(np.matrix(result_process_matrix).getH()))) \
            /np.sqrt(np.trace(np.matrix(ideal_process_matrix) @ np.array(np.matrix(ideal_process_matrix).getH())))

    def plot_detuning_fidelity(self, csv_path, save_path):
        from matplotlib import pyplot as plt
        df = pd.read_csv(csv_path)
        plt.plot(df['detuning'], df['fidelity'])
        plt.savefig(save_path)
        
    def frequency_sweep_tomography(self, base_gate, X_gate, file_name, save_dir = 'gate_tomography_result'):
        detuning_length = 10
        detuning_list = np.linspace(-2*self.Q_setup.MHz, +2*self.Q_setup.MHz, detuning_length)
        fidelity_list = np.empty(detuning_length)
        
        ideal_process_matrix = np.zeros((4, 4))
        ideal_process_matrix[1][1] = 1
        for idx, detuning in enumerate(detuning_list):
            pi_process_matrix = self.gate_tomography(X_gate, base_gate, detuning = detuning)
            fidelity_list[idx] = self.gate_fidelity(pi_process_matrix.reshape(4, 4), ideal_process_matrix)
        save_path = os.path.join(save_dir, file_name)
        df = pd.DataFrame({"detuning" : detuning_list, 'fidelity' : fidelity_list})
        df.to_csv(save_path)