# https://github.com/Sulfurous-Impersonation/QBreakout/blob/83be0b266fb29140ed2c3d538a1b4812b7ee79d5/assets/computer.py
import pygame
import qiskit

from . import globals
from qiskit.tools.monitor import job_monitor

class Computer:
    def __init__(self):
        pass
    def update(self):
        pass

class ClassicalComputer(Computer):
    def __init__(self, paddle, blocks):
        self.paddle = paddle
        self.blocks = blocks
        self.score = 0
        self.speed = 3

    def update(self, ball):
        if pygame.sprite.collide_mask(ball, self.paddle):
            ball.bounce()
        for i in self.blocks:
            if pygame.sprite.collide_mask(ball, i):
                ball.bounce()
                self.blocks.remove(i)

class QuantumComputer(Computer):
    def __init__(self, quantum_paddles, circuit_grid) -> None:
        qiskit.IBMQ.load_account() #load IBMQ account to run on QC
        self.paddles = quantum_paddles.paddles 
        self.score = 0
        self.circuit_grid = circuit_grid
        self.measured_state = 0
        self.last_measurement_time = pygame.time.get_ticks() - globals.MEASUREMENT_COOLDOWN_TIME

    def update(self, ball):
        current_time = pygame.time.get_ticks()
        # trigger measurement when the ball is close to quantum paddles
        if 88 < ball.rect.x / globals.WIDTH_UNIT < 92:
            if current_time - self.last_measurement_time > globals.MEASUREMENT_COOLDOWN_TIME:
                self.update_after_measurement()
                self.last_measurement_time = pygame.time.get_ticks()
        else:
            self.update_before_measurement()
    
        if pygame.sprite.collide_mask(ball, self.paddles[self.measured_state]):
            ball.bounce() 

    def update_before_measurement(self):
        simulator = qiskit.BasicAer.get_backend("statevector_simulator")
        circuit = self.circuit_grid.model.compute_circuit()
        transpiled_circuit = qiskit.transpile(circuit, simulator)
        statevector = simulator.run(transpiled_circuit, shots=100).result().get_statevector()

        for basis_state, amplitude in enumerate(statevector):
            self.paddles[basis_state].image.set_alpha(abs(amplitude)**2*255)

    def update_after_measurement(self):
        if not globals.QUANTUM:
            simulator = qiskit.BasicAer.get_backend("qasm_simulator")
            circuit = self.circuit_grid.model.compute_circuit()
            circuit.measure_all()
            transpiled_circuit = qiskit.transpile(circuit, simulator)
            counts = simulator.run(transpiled_circuit, shots=1).result().get_counts()
            self.measured_state = int(list(counts.keys())[0], 2)

        else:
            provider = qiskit.IBMQ.get_provider('ibm-q')
            qcomp = provider.get_backend('ibmq_quito')
            job = qiskit.execute(circuit, backend=qcomp)
            job_monitor(job)
            counts = job.result().get_counts()
            self.measured_state = int(list(counts.keys())[0], 2)


        for paddle in self.paddles:
            paddle.image.set_alpha(0)
        self.paddles[self.measured_state].image.set_alpha(255)
