# https://github.com/hornc/faf/blob/89300e2dabb50869c1fd2539745fdb2e7a30c9c4/faf.py
#!/usr/bin/env python3
import argparse
import sys
from qiskit import QuantumCircuit, execute
from qiskit import BasicAer
import matplotlib.pyplot as plt

version = '0.1α'
ABOUT = f"""
Finites at Fredy's esolang interpreter. (v{version})
https://esolangs.org/wiki/Finites_at_Fredy%27s
"""
DEBUG = False
SHOTS = 100

class Schedule:
    def __init__(self, data, storage):
        self.schedule = [night.strip() for night in data if night.strip() and night.strip()[0] != '#']
        self.storage = storage
        self.qc = QuantumCircuit(storage.size, storage.outsize)

    def circuit(self):
        """Create corresponding circuit."""
        # Init bits
        for i, b in enumerate(self.storage.locations):
            if '(' in self.storage.raw[i]:
                self.qc.initialize([0, 1] if b else [1, 0], i)
            elif b:
                self.qc.x(i)

        # Add CSWAPs
        for i, shift in enumerate(self.schedule):
            locs = [int(v) - 1 for v in shift.split(',')]
            if DEBUG:
                print(f'Night {i + 1}: {locs}')
            self.qc.cswap(locs[1], locs[0], locs[2])

        # Add Measurements
        q = [i for i, v in enumerate(storage.raw) if '!' in v]
        self.qc.measure(q[::-1], list(range(storage.outsize)))
        print(self.qc)

    def describe(self):
        """Describe the shifts as they unfold."""
        emp = 1
        print('\nSchedule\tStaff On Duty\tWest Coor.\tControl Room\tEast Coor.\tOutput (masked)')
        for i, shift in enumerate(self.schedule):
            t = [i + 1] + [int(v) for v in shift.split(',')]
            t += [self.storage.get(i) for i in t[1:]]
            t += [emp]
            r = self.storage.CSWAP(*t[1:4])
            t += [self.storage]
            print('Night {0}:\tEmployee #{7}\tLoc. {1} ({4})\tLoc. {2} ({5})\tLoc. {3} ({6})\t{8}'.format(*t))
            if r:
                emp += 1
        print()

    def draw(self):
        self.qc.draw(output='mpl')
        plt.show()

    def select(self):
        print('DEPRECATED: use storage.select() directly.')
        self.storage.select()

    def simulate(self):
        self.circuit()

        backend = BasicAer.get_backend('qasm_simulator')
        job = execute(self.qc, backend, shots=SHOTS)
        results = job.result().get_counts(self.qc)
        print('RESULTS (result: count):', results)
        r = list(results.keys())[0]
        print('As int:', int(r, 2))
        print('As chr:', chr(int(r, 2)))


class Storage:
    def __init__(self, data):
        self.input = ''
        self.raw = [v.strip() for v in data.split(',')]
        self.insize = data.count('(')
        self.locations = [None if 'EMPTY' in v or '(' in v else v for v in self.raw]
        self.size = len(self.locations)
        self.outmask = [i for i, v in enumerate(self.raw) if '!' in str(v)]
        self.outsize = len(self.outmask)

    def reset(self):
        """Reset locations to initial state."""
        self.locations = [None if 'EMPTY' in v or '(' in v else v for v in self.raw]
        self.select(self.input)

    def CSWAP(self, w, c, e):
        animatronics = (self.locations[w - 1], self.locations[e - 1])
        if self.locations[c - 1]:
            self.locations[w - 1], self.locations[e - 1] = animatronics[::-1]
            return True

    def get(self, i):
        """Get the contents of storage location i."""
        return self.locations[i - 1]

    def bits(self):
        """Output masked locations (!) as bits."""
        return ''.join(['1' if v else '0' for i, v in enumerate(self.locations) if i in self.outmask])

    def int(self):
        """Output masked locations (!) as an integer."""
        return int(self.bits(), 2)

    def chr(self):
        """Output masked locations (!) as a character."""
        return chr(self.int())

    def __repr__(self):
        return f"{self.bits()}: int: {self.int()} chr: {self.chr()}"


    def select(self, inputs=''):
        """
           Select optional animatronics.
           Prompts for user input.
        """
        if inputs:
            self.input = inputs[-self.insize:]
        for i, a in enumerate(self.raw):
            if '(' in a:
                if not inputs:
                    ans = input(f'Is {a.strip("!")} present? (y/N) ')
                    self.input += '1' if ans.upper() == 'Y' else '0'
                else:
                    ans = inputs[0]
                    inputs = inputs[1:]
                if ans.upper() in ('1', 'Y'):
                    self.locations[i] = a.strip('()!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=ABOUT, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('file', help='source file to process')
    parser.add_argument('--debug', '-d', help='turn on debug output', action='store_true')
    parser.add_argument('--circuit', '-c', help='display quantum circuit image', action='store_true')
    parser.add_argument('--bitstring', '-b', help='input for optional animatronics as a bitstring')
    parser.add_argument('--input', '-i', help='input for optional animatronics as an integer')
    args = parser.parse_args()

    DEBUG = args.debug
    with open(args.file, 'r') as f:
        storage = Storage(f.readline())
        schedule = Schedule(f.readlines(), storage)
    if DEBUG:
        print('Animatronics:', storage.raw, storage.locations)
        print('Schedule:', schedule.schedule)

    inputbits = args.bitstring or args.input and bin(int(args.input))[2:].rjust(storage.insize, '0')

    storage.select(inputbits)
    print('Input Animatronics:', schedule.storage.locations)
    schedule.describe()
    storage.reset()
    schedule.simulate()

    if args.circuit:
        schedule.draw()

