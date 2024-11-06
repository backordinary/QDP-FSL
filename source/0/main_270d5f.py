# https://github.com/grisuno/quantum/blob/2afef2c19b3a9a880fa9d255be93b1c8a78cc9b7/main.py
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

# Crea un circuito cuántico con dos qubits y dos bits clásicos
q = QuantumRegister(2)
c = ClassicalRegister(2)
circuit = QuantumCircuit(q, c)

# Aplica una puerta de Hadamard a ambos qubits para crear una superposición de estados
circuit.h(q[0])
circuit.h(q[1])

# Aplica una puerta de CNOT (puerta de control NOT) al segundo qubit, utilizando el primer qubit como control
circuit.cx(q[0], q[1])

# Mide ambos qubits y guarda los resultados en los bits clásicos
circuit.measure(q, c)

# Ejecuta el circuito y obtiene los resultados
backend = Aer.get_backend('qasm_simulator')
job = execute(circuit, backend, shots=1024)
result = job.result()
resultados = result.get_counts(circuit)

# Crea un gráfico de barras con los resultados
plt.bar(resultados.keys(), resultados.values())

# Añade etiquetas a los ejes y un título al gráfico
plt.xlabel('Resultado')
plt.ylabel('Frecuencia')
plt.title('Resultados del problema de los gemelos cuanticos')

# Muestra el gráfico
plt.show()
# Convierte los resultados en un dataframe de pandas
df = pd.DataFrame.from_dict(result.get_counts(circuit), orient='index')
df = df.rename(columns={0: 'frecuencia'})

# Calcula la frecuencia relativa y agrega una columna al dataframe
df['frecuencia_relativa'] = df['frecuencia'] / sum(df['frecuencia'])

# Muestra el dataframe
print(df)
