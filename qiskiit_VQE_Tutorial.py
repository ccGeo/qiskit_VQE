

######## Variational Quantum Eigensolver

# Based on IBM Quantum Learning  #
# see also arXiv:2111.05176

## VQE problems are excellent candidates for hybrid algorithms, incorporating quantum and
## Classical computation, on near-term noisy intermediate scale quantum (NISQ) devices.
# in principle they are based on the variational ansatz used to bound the groundstate energy of
# a given hamiltonian. E0<= <psi|H|psi>/<psi|psi>
# roughly the trial wavefunction |psi> can be decomposed as a set of unitary operators acting a
# register of qubits often initialized in the ground state |0>\equiv |0>^\otimes N where we paramertize the
# the set of unitaries as U(\theta), hence |psi>=U(\theta)|0>, the goal is then to optimize over
# theta and find the minimal energy eigenvalue. One maps the hamiltonian to a tensor product of Pauli
# operators referred to as pauli strings P_a\in {I,X,Y,Z}^{\otimes N}
# The hamiltonian then takes the form H=\sum_{a}^{p}w_a P_a, where we have w_a acting as a set of
# weights. The variational problem now gives
# E=min_\theta \sum_{a}^{p}w_a <0|U\dagger(\theta) P_a U(\theta)|0>
# here it should be clear the hybrid nature of the VQE. The expectation value can be computed
# on the quantum device, while the summation and minimization is computed on a conventional device.

# Lets follow the tutorial

import numpy as np
import matplotlib.pyplot as plt
import qiskit
# These are pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit import QuantumCircuit


# Classical reference state. Suppose we have a 3 qubit system which we want to start in the
# |001> state. This is an example of a purely classical reference state, to construct it, you simply
# need to apply a not gate (X) to the zero state of the first qubit. i.e. |001>=X_0|000>
# recall qubits are read from right to left.
qc=QuantumCircuit(3)
qc.x(0)

qc.draw(output="mpl",filename='qc.png')
#plt.show()
plt.close()

# now suppoer you want to start with a more complex state that includes superposition,
# such as 1/Sqrt[2] (|100>+|111>)
# one way to obtain this state is to apply a
# hadamard gate on q0, a cnot gate on q1 with q0 as the control cx(control,target) and
# an X gate on qubit two.
qc=QuantumCircuit(3)
qc.h(0)
qc.cx(0,1)
qc.x(2)
qc.draw("mpl")
#plt.show()
plt.close()

# We can also construct referencestates using template circuits.
# As an example. The TwoLocal
from qiskit.circuit.library import TwoLocal
from math import pi

reference_circuit=TwoLocal(2,"rx","cz",entanglement="linear",reps=1)
theta_list=[pi/2,pi/3,pi/3,pi/2]

reference_circuit=reference_circuit.assign_parameters(theta_list)
reference_circuit.decompose().draw("mpl")
#plt.show()
plt.close()

### There are also application-specific reference states
# Quantum Machnice Learning
# for variational quantum classifier (VQC), the training data is encoded into a quantum state
# with a parameterized circuit known as a feature map, each parameter value represents a data
# point from the training dataset. The ZZFeatureMap is a type of parameterized circuit that
# can be utilized to pass our data points (x) to this feature map.

from qiskit.circuit.library import ZZFeatureMap

data=[0.1,0.2]

zz_feature_map_reference=ZZFeatureMap(feature_dimension=2,reps=2)
zz_feature_map_reference=zz_feature_map_reference.assign_parameters(data)
zz_feature_map_reference.decompose().draw("mpl")
#plt.show()
plt.close()

###############################33
############### Parameterized Quantum Circuits

# Variational algorithms operate by exploring and comparaing a range of quantum states
# |\psi(\vec{\theta})>
#which depend on a finite set of k parameters \vec{\theta}=(\theta^0.....\theta^{k-1})
# These states can be prepared using a parameterized quantum circuit, where gates are defined with
# tunable parameters. It is possible to create this parameterized circuit without binding specific
# angles yet.

from qiskit.circuit import QuantumCircuit, Parameter

theta=Parameter("θ")
qc = QuantumCircuit(3)
qc.rx(theta, 0)
qc.cx(0, 1)
qc.x(2)

qc.draw("mpl")
#plt.show()
plt.close()

angle_list=[pi/3,pi/2]

circuits=[qc.assign_parameters({theta:angle}) for angle in angle_list]

for circuit in circuits:
    circuit.draw("mpl")
plt.close()

##############3 Variational Form and Ansatz

# To iteratively optimize from a refernce state |\rho> to a target state |\psi(\vec{\theta})>
# we need to define a variational form U_V(\vec{\theta}) that represents a collection of parameterized
#  states for our variational algorithm to explore:

# i.e.      |0> \rightarrow U_R|0> = |\rho> \rightarrow U_A(\vec{\theta})|0>
#                                    =U_V U_r |0>
#                                    =U_v |\rho>
#                                    =|\psi(\vec{\theta})>

# notice that \psi depends on the reference state as well as U_V, which contains parameters
# U_A is defined as U_A=U_V(\vec{\theta})U_R

# Notice dimensionality will quickly become a problem. An n-qubit system has an
# enormous number of possible states in its configuration space, D=2^{2n}
# and, the runtime complexity of search algorithms grows exponentially with dimension,
# this is the curse of dimensionality.
# Therefore, finiding an efficient truncated ansatz is an active area of research.
#

# An important example of heuristic ansatzes is the N-local circuits ansatz;
# from IBM Quantum Documentation:
### The structure of the n-local circuit are alternating rotation and entanglement layers.
### In both layers, parameterized circuit-blocks act on the circuit in a defined way.
### In the rotation layer, the blocks are applied stacked on top of each other, while
### in the entanglement layer according to the entanglement strategy. The circuit blocks can
### have arbitrary sizes (smaller equal to the number of qubits in the circuit).
### Each layer is repeated reps times, and by default a final rotation layer is appended.

# Essentially there are repeated layers of rotations and entanglement.
# each layer is formed by gates of size at most N, where N is lower then the # of qubits
# For rotation standard rotation operations like Rx are used
# for entanglement Toffoli or CX gates etc. used with a given strategy
# both types of layers can be parameterized.
# As an example:
# create a 5 qubit NLocal circuit with rotation blocs formed by Rx and CRZ gates
# entanglement blocks formed by Toffoli gates act on [0,12], [0,2,3] [4,2,1] and [3,1,0]
# with two repitions per layer.

from qiskit.circuit.library import NLocal, CCXGate, CRZGate, RXGate
from qiskit.circuit import Parameter

theta = Parameter("θ")
ansatz= NLocal(
    num_qubits=5,
    rotation_blocks=[RXGate(theta), CRZGate(theta)],
    entanglement_blocks=CCXGate(),
    entanglement=[[0,1,2],[0,2,3],[4,2,1],[3,1,0]],
    reps=2,
    insert_barriers=True,
)
ansatz.decompose().draw("mpl")
#plt.show()
plt.close()

# in the circuit, the Toffoli gate is the largest gate, acting on 3 qubits. Therefore the circuit
# is 3-local. The most common type is 2-local, with single qubit rotation and 2-qubit entanglement
# there is a dedicated 2-local class.
# the sytanx is roughly the same.
# but the layers can be passed as strings and there is no need for importing a parameter definition

from qiskit.circuit.library import TwoLocal

ansatz=TwoLocal(
    num_qubits=5,
    rotation_blocks=["rx","rz"],
    entanglement_blocks="cx",
    entanglement="linear",
    reps=2,
    insert_barriers=True,
)

ansatz.decompose().draw("mpl")
#plt.show()
plt.close()
# more about twolocal can be found here
# # https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.TwoLocal

# EfficientSU2
# This is a hardware-efficient circuit that consisters of layers of single-qubit operations
# spanning SU(2) and CX entanglements. It is a heuristic pattern that can be used to
# prepare trial wave functions for variational quantum algorithms or classicification circuit for
# machine learning.

from qiskit.circuit.library import EfficientSU2

ansatz=EfficientSU2(4, su2_gates=["rx","y"], entanglement="linear", reps=1)
ansatz.decompose().draw("mpl")
#plt.show()
plt.close()

# it is often useful to apply problem specific knowlede to restrict our circuit search space
# to a specific type. This helps us gain speed without losing accuracy.

#Optimization
# In a max-cut problem, we wnat to partiion nodes of a graph in a way that maximizes the number
# of edges between nodes in differing groups. The desired max-cut partition for the graph below is clear:
# The node on the left should be spearated from the rest of the nodes on teh right by a cut.

import rustworkx as rx
from rustworkx.visualization import mpl_draw

n=4
G=rx.PyGraph()
G.add_nodes_from(range(n))
#The syntax is (start,end,weight)
edges=[(0,1,1.0),(0,2,1.0),(0,3,1.0),(1,2,1.0),(2,3,1.0)]
G.add_edges_from(edges)

mpl_draw(G,pos=rx.shell_layout(G),with_labels=True,edge_labels=str, node_color="#1192E8")
plt.close()

# qiskit allows for easy implementation of quantum optimatization algorithms
# here we will use the Quantum Approximate Optimaization Algorithm QAOA (see arXiv:1411.4028 for more details)
# This was very briefly better then any know polynomial time classical algorithm
# see arXiv:1505.03424 for more information about the better classical algorithm.
# From Wikipedia: QAOA consists of
# 1. Defining a cost hamiltonian H_c such that its ground state encodes the solution to the problem
# 2. Defining a mixer Hamiltonian H_M
# 3. Defining oracles (unitary time evolution) U_C(\gamma)=exp(-i\gamma H_C), and
#    U_M(\alpha)=exp(-i\alpha H_M)
# 4. Repeated application of the oracles (time evolution) U_C and U_M in the order
#    U(\vec{\gamma},\vec{\alpha}) = \prod_{i=1}^N U_M(\gamma_i)U_M(\alpha_i)
# 5. Preparing an initial state, that is a superposition of all possible staes and applying U(\vec{\gamma},\vec{\alpha})
# 6. Using classical methods to optimize \vec{\gamma},\vec{\alpha}
#
# To utilize it we need a Pauli Hamiltonian that encodes the cost in a manner s.t. the min expectation
# value of the operator corresponds to the maximum number of edges between nodes in two different groups.
#
# For this simple example the operator is alinear combination of terms with Z operators on nodes
# connected by an edge ZZII+IZZI+ZIIZ+IZIZ+IIZZ. Once the operator is construct the ansatz for the
# QAOA algorithm can easily be built by using QAOAAnsatz

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

# Problem to Hamiltonian operator.
hamiltonian =SparsePauliOp.from_list([("ZZII",1),("IZZI",1),("ZIIZ",1),("IZIZ",1),("IIZZ",1)])
#QAOA ansatz circuit
ansatz=QAOAAnsatz(hamiltonian,reps=2)
#draw
ansatz.decompose(reps=3).draw("mpl")
#plt.show()
plt.close()
# the circuit is drawn in all its glory from basic gates. But using a different command we
# can gain a more high level look.
ansatz.draw("mpl")
#plt.show()
plt.close()
# or potentially more useful
ansatz.decompose(reps=2).draw("mpl")
plt.close()

# Quantum Machine Learning

# in machine learning, a common application is the classification of data into two or more categories.
# This involvesencoding a datapoint into a feature map that maps classical feature vectors
# into the quantum hilbert space. Constructing quantum feature maps based onparameterized
# quantum circuits that are hard to simulate classically is an important step towards obtaining
# a potential advantage over classical machine learning approaches and is an active area of
# current research.

# The ZZFeatureMap can be used to create a parameterized circuit. We can pass in our data points
# to the feature map (x) and a speratre variantional form to pass in weights as parameters (\theta)

from qiskit.circuit.library import ZZFeatureMap, TwoLocal

data=[0.1,0.2]

zz_feature_map_reference= ZZFeatureMap(feature_dimension=2,reps=2)
zz_feature_map_reference=zz_feature_map_reference.assign_parameters(data)

variation_form=TwoLocal(2,["ry","rz"],"cz",reps=2)
vqc_ansatz=zz_feature_map_reference.compose(variation_form)
vqc_ansatz.decompose().draw("mpl")
plt.close()

# This part of the tutorial has displayed how to define the search space with a variational form.

######################################################################

############3 Cost Functions

# During this less we will learn how to evaluate a cost function.
# 1. Learn qiskit runtime primitives
# 2. Defining a cost function.
# 3. Defining a measurement strategy with the primitives to optimize speed vs accuracy

### 1. Primitives

##### Qiskit offers two primitives that can help measure a quantum system
# 1. Sampler: Given a quantum state|psi> this primitive obtains the probability of each possible
#             computational basis state.
# 2. Estimator: Given an observable H, and state |psi>, this primitive computes the
#              expectation value of H in the state |psi> i.e. <psi|H|psi>

### Sampler:
# The smapler primitive calculates the probability of obtaining each possible state |k>
# from the computational basis, given a quantum circuit that prepares the state |psi>
# p_k=|<k|psi>|^2 \forall k \in \mathbb{Z}_2^n={0,1,.... 2^n-1} where n is the number of qubits.
# here k is the interger represetation of any possible output binary string {0,1}^n (i.e. integers base 2)

# Qiskit Runtime's Sampler runs the circuit multiple times on a quantum device, performing
# measurements on each run, and reconstructing the probability distribution from the recovered
# string bits. The more runs (or shots) it performs, the more accurate the results will be,
# but this requires more time and resources.

# Since the # of possible outputs grows exponentially with the number of qubits (2^n)
# the number of shots will need to grow exponentially as well in order to capture a dense
# proability distribution. Therefore, sampler is only eifficient for sparse probability distributionsl
# where the target state |psi> must be expressible as a linear combination of the computational
# basis staes, with teh number of terms growing at most polynomially with the number of qubits.
# |psi> = \sum_k^{poly(n)} w_k |k>
# Note: sampler can be programed to only sample a subset of the circuit, returning a subset of
# the possible states.

### Estimator
# The esitmator primitive calculate shte expectation value of an obserable H for a state |psi>
# where the observable probabilities can be expressed as p_\lambda=|<\lambda|\psi>|^2, where
# |\lambda> is an eigenstate of the observable. The expectation value is then defined as the average of
# all possible outcomes \lambda of a measurement of the state |psi>, weighted by the
# corresponding probabilities. <H>_psi= \sum_\lambda p_\lambda \lambda = <\psi|H|\psi>
# However, calculation the expectation value of an aobservable is not always possible,
# as we often don't know its eigenbasis. Qiskit Runtime's Estimator use a complex algebraic process to
# estimate the expectation value on a real quantum device by breaking down the observable
# into a combination of other observables whose eigenbasis is know.
#
# Estimator breaks down any observables that it doesn't know how to measure into simpler
# measureable observables, the Pauli operators.

# Any operator can be expressed as a combination of the 4^n Pauli operators
# P_k =\sigma_{k_{n-1}} \otimes \cdots \otimes \sigma_{k_0} \forall k \in \mathbb{Z}^n_4
# such that H=\sum_{k=0}^{4^n-1} w_k P_k
# where n is the number of qubits, k=k_{n-1}\cdots k_0 for k_l\in \in \mathbb{Z}_4={0,1,2,3}
# and \sigma_i=(\sigma_0,...,\sigma_3)=(I,X,Y,Z)

# After performing this decomosition , Estimator derives a new circuit V_k|psi> for each P_l
# to effectively diagonalize the Pauli observable in the computational basis and measure it.
# We can easly measure the Pauli observables since we know V_k ahead of time, which is not
# the case in general for arbitrary observables.
# For each P_k, the estimator runs the corresponding circuit ona quantum device multiple times,
# measures the output state in the computational basis, and calculates the probability p_{k_j} of
# obtaining each possible output j. it then looks for the eigenvalue \lambda_{k_j} of P_k corresponding
# to each output j, multiples by w_k and adds all the results together to obtain the expected value
# of H for a given state |psi>
# <H>_\psi=\sum_{k=0}^{4^n-1}w_k \sum_{j=0}^{2^n-1}p_{k_j}\lambda_{k_j}

# Since calculating <H> for 4^n Pauli's is impractical, estimator can only be efficient
# when n is small or when a large number of the w_k are zero (i.e. a sparse Pauli decomposition)
# Formally, for this to be efficiently solvable, the number of non-zero terms has to grow as most
# polynomially with the number of qubits n. Notice this is also true of the sampling,
# hence we need  <H>_\psi=\sum_{k=0}^{Poly(n)}w_k \sum_{j=0}^{Poly(n)}p_{k_j}\lambda_{k_j}

# Guided example with expectation values
# lets assume a single qubit |+>=\frac{1}{\sqrt{2}} (|0>+|1>)=H|0> (H is hadamard)
# and \hat{H}=\begin{pmatrix} -1 & 2 \\ 2 & 1 \\ \end{pmatrix}
#            = 2X-Z
#with the theoretical value <\hat{H}>_+=2

# Since we do not know how to compute this observable directly, we need to reexpress it
# as <H>_+= 2<X>_+ - <Z>_+
# Lets compute X and Z direclty, notice, X and Z do not commute,
# i.e. they are not mutually commuting observables, therefore,
# they cannot be simultaneously measured. We will need to auxillary circuits to do this.

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# Note: The following code will work for any other initial
# single-qubit state and observable
original_circuit = QuantumCircuit(1)
original_circuit.h(0)

H=SparsePauliOp(["X","Z"],[2,-1])

aux_circuits=[]
for pauli in H.paulis:
    #copy the original circuit
    aux_circ=original_circuit.copy()
    aux_circ.barrier()
    if str(pauli) == "X":
        aux_circ.h(0)
    elif str(pauli) == "Y":
        aux_circ.sdg(0)
        #Single qubit S-adjoint gate (~Z**0.5).
        # It induces a −π/2 phase.
        # This is a Clifford gate and a square-root of Pauli-Z.
        aux_circ.h(0)
        # h is a hadamard gate.
    else:
        aux_circ.id(0)
    aux_circ.measure_all()
    aux_circuits.append(aux_circ)

original_circuit.draw("mpl")
for circuit in aux_circuits:
    circuit.draw("mpl")
plt.close()

# the first is the auxiliary circuit for X
# the second is the auxiliary circuit for Z.

# Now, we carry out the computation using sampler and check the results using estimator
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.result import QuasiDistribution
from qiskit.circuit.library import IGate, ZGate, XGate, YGate
import numpy as np

## The sampler portion
shots=10000
# we will sample 10000 times
sampler=StatevectorSampler()
job=sampler.run(aux_circuits,shots=shots)
data_pub=job.result()[1].data
bitstrings=data_pub.meas.get_bitstrings()
counts=data_pub.meas.get_counts()
quasi_dist = QuasiDistribution({outcome: freq / shots for outcome, freq in counts.items()})

expvals=[]
for pauli in H.paulis:
    val=0

    if str(pauli) == "I":
        Lambda=IGate().to_matrix().real

    if str(pauli) == "X":
        Lambda = XGate().to_matrix().real
        val += Lambda[0][1]*quasi_dist.get(1)
        val += Lambda[1][0]*quasi_dist.get(0)

    if str(pauli) == "Y":
        Lambda = YGate().to_matrix().real
        val += Lambda[0][1]*quasi_dist.get(1)
        val += Lambda[1][0]*quasi_dist.get(0)
        #notice the tutorial uses XGate with 1.j and -1.j , why not just use YGate?

    if str(pauli) == "Z":
        Lambda = ZGate().to_matrix().real
        val += Lambda[0][0]*quasi_dist.get(1)
        val += Lambda[1][1]*quasi_dist.get(0)

    expvals.append(val)

# here we make use of f-strings, think of it as ToString in mathematica
# zip, which is identical to Thread[] in mathematica except it acts on tuples, immutable lists
# and for numerical values we use num:.nf where n parameterizes the number of sig figs we want.
print("Sampler results:")
for (pauli, expval) in zip(H.paulis,expvals):
    print(f" >> Expected value: {str(pauli)}: {expval:.5f}")

total_expval= np.sum(H.coeffs * expvals).real
print(f" >> Total expected value: {total_expval:.5f}")


###### Estimator
obserables = [
    *H.paulis,H
] # Note: run for individual paulis as well as for the full observable to compare to above

estimator=StatevectorEstimator()
job=estimator.run([(original_circuit,obserables)])
estimator_expvals=job.result()[0].data.evs

print("Estimator Results:")
for (obs,expval) in zip(obserables,estimator_expvals):
    if obs is not H:
        print(f" >> Expected value of {str(obs)}: {expval:.5f}")
    else:
        print(f" >> Total expected value: {expval:.5f}")

# notice, this is a sample of the distriubtion, run the code again, you will get different results
# from the sampler, as expected.

# Notice the mistake/bad notation in the Mathematical rigor section...
# the author should introduce the kronecker delta such that <j|Lambda|k> = lambda_j \delta_{jk}
# then from the third to fourth line insert this.. <j|Lambda|k> = lambda_j \delta_{jk}
# then compute the sum,
# leaving a single summation and \lambda_j outfront


