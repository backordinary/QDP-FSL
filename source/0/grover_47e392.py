# https://github.com/PetkaRedka/Kvanti/blob/76050e2025a6e45d66d84054fa1bb438c240a42c/Grover.py
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister

# Уравнение: 2 - x = -1

# Три регистра - для первого числа, второго и их разности
A, B, C = QuantumRegister(2, name = "a"), QuantumRegister(2, name = "b"), QuantumRegister(3, name = "c")
# Вспомогательныерегистры для переходов и определения знака
Support = QuantumRegister(3, name = "support")
# Регистр ответа
Check_Out = QuantumRegister(1, name = "check_out")
# Два классических регистра для для классических битов 
cr1 = ClassicalRegister(1)
cr2 = ClassicalRegister(1)
Grover_circuit = QuantumCircuit(A, B, C, Support, Check_Out, cr1, cr2)

# Объявляем симулятор
simulator = Aer.get_backend('qasm_simulator')

# Переведем кубиты в состояние суперпозиции
Grover_circuit.h(A[0])
Grover_circuit.h(A[1])
Grover_circuit.barrier()

# Инициализируем B как 2
Grover_circuit.initialize([1, 0], B[0])
Grover_circuit.initialize([0, 1], B[1])
Grover_circuit.barrier()

# Oracle
# Посчитаем младший кубит
Grover_circuit.cx(A[0], C[0]) 
Grover_circuit.cx(B[0], C[0]) 

# Проверка на отрицательность
Grover_circuit.x(B[0])
Grover_circuit.ccx(A[0], B[0], Support[0])
Grover_circuit.x(B[0])
Grover_circuit.barrier()

# Считаем старший кубит
Grover_circuit.cx(A[1], C[1])
Grover_circuit.cx(B[1], C[1])
Grover_circuit.x(B[1])
Grover_circuit.ccx(A[1], B[1], Support[1]) 
Grover_circuit.x(B[1])
Grover_circuit.barrier()

# Меняем значащий кубит, если число отрицательное
Grover_circuit.x(C[1])
Grover_circuit.ccx(Support[0], C[1], Support[2])
Grover_circuit.x(C[1])
# Применяем оператор ИЛИ
Grover_circuit.x(Support[1])
Grover_circuit.x(Support[2])
Grover_circuit.x(C[2])
Grover_circuit.ccx(Support[1], Support[2], C[2]) 
Grover_circuit.x(Support[1])
Grover_circuit.x(Support[2])
Grover_circuit.barrier()

# Приводим число в обратный код
Grover_circuit.cx(Support[1], C[1])
Grover_circuit.cx(Support[0], C[0])
Grover_circuit.barrier()

# Проверяем его с -1 в обратном коде
Grover_circuit.mct(C, Check_Out)
Grover_circuit.barrier()

# Uncomputing
Grover_circuit.cx(Support[1], C[1])
Grover_circuit.cx(Support[0], C[0])
Grover_circuit.barrier()
Grover_circuit.x(Support[2])
Grover_circuit.x(Support[1])
Grover_circuit.ccx(Support[1], Support[2], C[2]) 
Grover_circuit.x(C[2])
Grover_circuit.x(Support[2])
Grover_circuit.x(Support[1])
Grover_circuit.x(C[1])
Grover_circuit.barrier()
Grover_circuit.x(B[1])
Grover_circuit.ccx(A[1], B[1], Support[1])
Grover_circuit.x(B[1])
Grover_circuit.cx(B[1], C[1])
Grover_circuit.cx(A[1], C[1])
Grover_circuit.barrier()
Grover_circuit.x(B[0])
Grover_circuit.ccx(A[0], B[0], Support[0])
Grover_circuit.x(B[0])
Grover_circuit.cx(B[0], C[0])
Grover_circuit.cx(A[0], C[0])
Grover_circuit.barrier()

# Reflection
Grover_circuit.h(A[0])
Grover_circuit.h(A[1])
Grover_circuit.z(A[0])
Grover_circuit.z(A[1])

Grover_circuit.cz(A[0],A[1])

Grover_circuit.h(A[0])
Grover_circuit.h(A[1])

Grover_circuit.measure(A[0], cr1[0])
Grover_circuit.measure(A[1], cr2[0])

# Симуляция
job = execute(Grover_circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts(Grover_circuit)
# Результаты симуляции
print(counts)

print(Grover_circuit.draw())