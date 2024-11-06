# https://github.com/dsaf2007/QuantumComputing/blob/63e87442e356dd48d3d35018cbb225964329fee2/project1/backup.py
#################################
# Quantum Battleship from tutorial @
# https://github.com/QISKit/qiskit-tutorial
##################################
import sys
# Checking the version of PYTHON; we only support > 3.5
if sys.version_info < (3, 5):
 raise Exception('Please use Python version 3.5 or greater.')

from qiskit import QuantumProgram
import Qconfig
import getpass
import random
import numpy
import math

## 1. Select a backend: IBM simulator (ibmqx_qasm_simulator) or real chip ibmqx2

d = input("Do you want to play on the real device? (y/n)\n").upper()
if (d == "Y"):
  device = 'ibmqx2'
else:
  device = 'ibmqx_qasm_simulator'
# note that device should be 'ibmqx_qasm_simulator', 'ibmqx2' or 'local_qasm_simulator'
# while we are at it, let's set the number of shots
shots = 1024
APItoken = 'YOUR API TOKEN'
config = {
    'url': 'https://quantumexperience.ng.bluemix.net/api',
}


####### 2. players to set up their boards.
randPlace = input("> Press Enter to start placing ships...\n").upper()

# The variable ship[X][Y] will hold the position of the Yth ship of playerX+1
shipPos = [[-1]*3 for _ in range(2)]

# loop over both players and all three ships for each
for player in [0, 1]:
  # if we chose to bypass player choice and do random, we do that
  if ((randPlace == "r") | (randPlace == "R")):
    randPos = random.sample(range(5), 3)
    for ship in [0, 1, 2]:
      shipPos[player][ship] = randPos[ship]
  else:
    for ship in [0, 1, 2]:
    # ask for a position for each ship,
      choosing = True
      while (choosing):
        # get player input
        position = getpass.getpass("Player " + str(player+1)
                            + ", choose a position for ship " + str(ship+1) +
                            " (0-4)\n")
        # see if the valid input and ask for another if not
        if position.isdigit():  # valid answers have to be integers
          position = int(position)
    # they need to be between 0 and 5
          if (position in [0, 1, 2, 3, 4]) and (not position in shipPos[player]):
            shipPos[player][ship] = position
            choosing = False
            print("\n")
          elif position in shipPos[player]:
            print("\nYou already have a ship there. Try again.\n")
          else:
            print("\nThat's not a valid position. Try again.\n")
        else:
          print("\nThat's not a valid position. Try again.\n")

 ########### 3. Main loop.
# Each iteration starts by asking players where on the opposing grid theywant a bomb.
# The quantum computer calculates the effects of the bombing, and theresults are presented.
# The game continues until all the ships of one player are destroyed.
game = True

# the variable bombs[X][Y] holds the number of times position Y has been bombed by player X+1
bomb = [[0]*5 for _ in range(2)]  # all values are initialized to zero

# the variable grid[player] will hold the results for the grid of each
player
grid = [{}, {}]
while (game):
  input("> Press Enter to place some bombs...\n")
  print("\n\nIt's now Player " + str(0+1) + "'s turn.\n")
  # keep asking until a valid answer is given
  choosing = True
  while (choosing):
    # get player input
    position = input("Choose a position to bomb (0, 1, 2, 3 or 4)\n")
    # see if this is a valid input. ask for another if not
    if position.isdigit():  # valid answers have to be integers
      position = int(position)
      if position in range(5):
        bomb[0][position] = bomb[0][position] + 1
        choosing = False
        print("\n")
      else:
        print("\nThat's not a valid position. Try again.\n")
    else:
      print("\nThat's not a valid position. Try again.\n")
  print("\n\nIt's now Player " + str(1+1) + "'s turn.\n")
  # keep asking until a valid answer is given
  choosing = True
  while (choosing):
    # get player input
    position = input("Choose a position to bomb (0, 1, 2, 3 or 4)\n")
    # see if this is a valid input. ask for another if not
    if position.isdigit():  # valid answers have to be integers
      position = int(position)
      if position in range(5):
        bomb[1][position] = bomb[1][position] + 1
        choosing = False
        print("\n")
      else:
        print("\nThat's not a valid position. Try again.\n")
    else:
      print("\nThat's not a valid position. Try again.\n")
  if device == 'ibmqx2':
    print("\nUsing a quantum computer for Player " + str(0+1) + "'s ships.\n")
  else:
    print("\nUsing the simulator for Player " + str(0+1) + "'s ships.\n")
  if device == 'ibmqx2':
    print("\nUsing a quantum computer for Player " + str(1+1) + "'s ships.\n")
  else:
    print("\nUsing the simulator for Player " + str(1+1) + "'s ships.\n")


 # now to set up the quantum program (QASM) to simulate the grid for this player
Q_program = QuantumProgram()
# set the APIToken and API url
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"])
# declare register of 5 qubits
q = Q_program.create_quantum_register("q", 5)
# declare register of 5 classical bits to hold measurement results
c = Q_program.create_classical_register("c", 5)
# create circuit
gridScript = Q_program.create_circuit("gridScript", [q], [c])
# add as many bombs as have been placed at this position
for n in range(bomb[(player+1) % 2][0]):
  # the effectiveness of the bomb
  # (which means the quantum operation we apply)
  # depends on which ship it is
  for ship in [0, 1, 2]:
    if (0 == shipPos[player][ship]):
      frac = 1/(ship+1)
      # add this fraction of a NOT to the QASM
      gridScript.u3(frac * math.pi, 0.0, 0.0,q[0])
# add as many bombs as have been placed at this position
for n in range(bomb[(player+1) % 2][1]):
  # the effectiveness of the bomb
  # (which means the quantum operation we apply)
  # depends on which ship it is
  for ship in [0, 1, 2]:
    if (1 == shipPos[player][ship]):
      frac = 1/(ship+1)
      # add this fraction of a NOT to the QASM
      gridScript.u3(frac * math.pi, 0.0, 0.0,q[1])
# add as many bombs as have been placed at this position
for n in range(bomb[(player+1) % 2][2]):
  # the effectiveness of the bomb
  # (which means the quantum operation we apply)
  # depends on which ship it is
  for ship in [0, 1, 2]:
    if (2 == shipPos[player][ship]):
      frac = 1/(ship+1)
      # add this fraction of a NOT to the QASM
      gridScript.u3(frac * math.pi, 0.0, 0.0,q[2])
# add as many bombs as have been placed at this position
for n in range(bomb[(player+1) % 2][3]):
  # the effectiveness of the bomb
  # (which means the quantum operation we apply)
  # depends on which ship it is
  for ship in [0, 1, 2]:
    if (3 == shipPos[player][ship]):
      frac = 1/(ship+1)
      # add this fraction of a NOT to the QASM
      gridScript.u3(frac * math.pi, 0.0, 0.0,q[3])
# add as many bombs as have been placed at this position
for n in range(bomb[(player+1) % 2][4]):
  # the effectiveness of the bomb
  # (which means the quantum operation we apply)
  # depends on which ship it is
  for ship in [0, 1, 2]:
    if (4 == shipPos[player][ship]):
      frac = 1/(ship+1)
      # add this fraction of a NOT to the QASM
      gridScript.u3(frac * math.pi, 0.0, 0.0,q[4])
gridScript.measure(q[0], c[0])
# to see what the quantum computer is asked to do, we can print the QASM file
# this lines is typically commented out
#print( Q_program.get_qasm("gridScript") )
# compile and run the QASM
results = Q_program.execute(["gridScript"], backend=device, shots=shots)

# extract data
grid[player] = results.get_counts("gridScript")
gridScript.measure(q[1], c[1])
# to see what the quantum computer is asked to do, we can print the QASM file
# this lines is typically commented out
#print( Q_program.get_qasm("gridScript") )
# compile and run the QASM
results = Q_program.execute(["gridScript"], backend=device, shots=shots)

# extract data
grid[player] = results.get_counts("gridScript")
gridScript.measure(q[2], c[2])
# to see what the quantum computer is asked to do, we can print the QASM file
# this lines is typically commented out
#print( Q_program.get_qasm("gridScript") )
# compile and run the QASM
results = Q_program.execute(["gridScript"], backend=device, shots=shots)

# extract data
grid[player] = results.get_counts("gridScript")
gridScript.measure(q[3], c[3])
# to see what the quantum computer is asked to do, we can print the QASM file
# this lines is typically commented out
#print( Q_program.get_qasm("gridScript") )
# compile and run the QASM
results = Q_program.execute(["gridScript"], backend=device, shots=shots)

# extract data
grid[player] = results.get_counts("gridScript")
gridScript.measure(q[4], c[4])
# to see what the quantum computer is asked to do, we can print the QASM file
# this lines is typically commented out
#print( Q_program.get_qasm("gridScript") )
# compile and run the QASM
results = Q_program.execute(["gridScript"], backend=device, shots=shots)

# extract data
grid[player] = results.get_counts("gridScript")


# we can check up on the data if we want
# these lines are typically commented out
#print( grid[0] )
#print( grid[1] )
# if one of the runs failed, tell the players and start the round again
if (('Error' in grid[0].values()) or ('Error' in grid[1].values())):
  print("\nThe process timed out. Try this round again.\n")
else:
  # look at the damage on all qubits (we'll even do ones with no ships)
  # # this will hold the prob of a 1 for each qubit for each player
  damage = [[0]*5 for _ in range(2)]
  for bitString in grid[0].keys():
    # and then over all positions
    for position in range(5):
      # if the string has a 1 at that position, we add a contribution to the damage
      # remember that the bit for position 0 is the rightmost one, and so at bitString[4]
      if (bitString[4-position] == "1"):
        damage[0][position] += grid[0]
        [bitString]/shots
  for bitString in grid[1].keys():
    # and then over all positions
    for position in range(5):
      # if the string has a 1 at that position, we add a contribution to the damage
      # remember that the bit for position 0 is the rightmost one, and so at bitString[4]
      if (bitString[4-position] == "1"):
        damage[1][position] += grid[1]
        [bitString]/shots


# give results to players
for player in [0, 1]:
  input("\nPress Enter to see the results for Player" + str(player+1) + "'s ships...\n")
  # report damage for qubits that are ships, with significant damage
  # ideally this would be non-zero damage,
  # so we choose 5% as the threshold
  display = [" ? "]*5

  # loop over all qubits that are ships
  for position in shipPos[player]:
    # if the damage is high enough, display the damage
    if (damage[player][position] > 0.1):
      if (damage[player][position] > 0.9):
        display[position] = "100%"
      else:
        display[position] = str(int(100*damage[player][position])) + "% "
       
  print("Here is the percentage damage for ships that have been bombed.\n")
  print(display[4] + " " + display[0])
  print(" |\ /|")
  print(" | \ / |")
  print(" | \ / |")
  print(" | " + display[2] + " |")
  print(" | / \ |")
  print(" | / \ |")
  print(" |/ \|")
  print(display[3] + " " + display[1])
  print("\n")
  print("Ships with 95% damage or more have been destroyed\n")
  print("\n")
  # if a player has all their ships destroyed, the game is over
  # ideally this would mean 100% damage, but we go for 90% because of noise again
  if (damage[player][shipPos[player][0]] > .9) and (damage[player][shipPos[player][1]] > .9) and (damage[player][shipPos[player][2]] > .9):
    print("***All Player " + str(player+1) + "'s ships have been destroyed!** *\n\n")
    game = False
  if (game is False):
    print("")
    print("=======GAME OVER=======")
    print("")
