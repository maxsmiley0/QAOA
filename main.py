import math

from IPython.core.display_functions import display
from qiskit import *
import numpy as np
from qiskit.visualization import plot_histogram
import qiskit.tools.jupyter
from qiskit.tools.monitor import job_monitor
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.circuit.library.standard_gates import PhaseGate, RXGate
import matplotlib.pyplot as plt
import random
"""
n = 4
beta = math.pi / 4
print(beta)

not_gate = np.array([[0, 1], [1, 0]])
id_gate = np.array([[1, 0], [0, 1]])

def B(n):
    #Overall matrix
    matrix = None

    #identity tensored together with not in index i
    for not_i in range(n):
        matrix_i = None
        for tensor in range(n):
            gate = None
            if not_i == tensor:
                gate = not_gate
            else:
                gate = id_gate
            if matrix_i is None:
                matrix_i = gate
            else:
                matrix_i = np.kron(matrix_i, gate)
        if matrix is None:
            matrix = matrix_i
        else:
            matrix = matrix + matrix_i
    return matrix

def mixer(n):
    return np.exp(-1j * beta * B(n))

qc_x = QuantumCircuit(n, n)
for i in range(n):
    qc_x.rx(2 * beta, i)

qc_y = QuantumCircuit(n, n)
for i in range(n):
    qc_y.ry(2 * beta, i)

print(qc_x.draw())
print(qc_y.draw())

"""

def get_Clauses(name):
    f = open(name)
    f_prime = f.read()
    str_lines = f_prime.split("\n")
    #Matrix of lists based on command structure.
    clauses = []
    line_split = []
    for line in str_lines:
        line_split = line.split(" ")
        if line_split[0] != "":
            if line_split[0] == "p":
                #Number of bits
                num_variables = int(line_split[2])
                #Number of clauses
                num_clauses = int(line_split[3])
            elif line_split[0] != "c":
                clause = []
                #First entry of  all clauses is the weight, as placed in the sheet.
                i = 0
                for instruct in line_split:
                    if i == 0:
                        clause += [float(instruct)]
                        i += 1
                    else:
                        clause += [int(instruct)]
                clauses += [clause]
    return clauses, num_variables, num_clauses


def get_Separator(clauses, v, c, gamma):
    qc = QuantumCircuit(v + c)
    clause_index = 0
    for clause in clauses:
        i = 0
        weight = 0
        not_list = []
        circuit_instruct = []
        for term in clause:
            if i == 0:
                i += 1
            elif term > 0:

                # We need to do the opposite so that only the all 0's case is marked as one. This will be flipped at the
                # end to get the total or on the output bit.
                circuit_instruct += [term - 1]
                not_list += [term - 1]
                qc.x(term - 1)
            elif term < 0:
                circuit_instruct += [abs(term) - 1]
        circuit_instruct.sort()
        qc.mct(circuit_instruct, v + clause_index)
        for was_notted in not_list:
            qc.x(was_notted)
        # Flips the last bit to allow for the system to be marked as 1 for everything by all 0's
        qc.x(v + clause_index)
        custom_gamma = PhaseGate(-gamma).control(num_ctrl_qubits=1)
        for target in circuit_instruct:
            qc.append(custom_gamma, [v + clause_index, target])
        qc.barrier(list(range(v + c)))
        clause_index += 1
    display(qc.draw("mpl"))
    qc = RemoveBarriers()(qc)
    U_Sep = qc.to_gate()
    U_Sep.name = "Usep"
    return U_Sep


clauses = []
v = 0
c = 0
clauses, v, c = get_Clauses("instructions.txt")
print(clauses, v, c)
gamma = np.pi / 6
beta = np.pi / 6
separator = get_Separator(clauses, v, c, gamma)
print(separator)

q = QuantumRegister(6)
c = ClassicalRegister(6)
qc = QuantumCircuit(q, c)
for i in range(6):
    qc.h(i)

for i in range(100):
    qc.append(separator, q)
    for i in range(6):
        qc.rx(math.pi / 3, i)

qc.measure_all()


#print(qc.draw())

aer_sim = Aer.get_backend('aer_simulator')
shots = 1
t_qpe = transpile(qc, aer_sim)
qobj = assemble(t_qpe, shots=shots)
results = aer_sim.run(qobj).result()
answer = results.get_counts()

print(answer)

print(clauses)

def get_literal_set(bitstring):
    literal_set = set()
    for i in range(len(bitstring)):
        if bitstring[i] == "0":
            literal_set.add("-" + str(i + 1))
        else:
            literal_set.add(str(i + 1))
    return literal_set

def get_score(bitstring):
    score = 0
    literal_set = get_literal_set(bitstring)
    for clause in clauses:
        weight = clause[0]
        vars = clause[1:]
        for var in vars:
            if str(var) in literal_set:
                score += weight
                break
    return score

print(get_score("010"))