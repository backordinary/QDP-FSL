# https://github.com/Jpifer13/qiskit_computing/blob/84b42fd06bdc6089c5c0c3af5b50095c1153f141/app.py
from application import create_app
import qiskit

App = create_app()

if __name__ == '__main__':
    print(f"Qiskit versions: {qiskit.__qiskit_version__}")
    App.run()
