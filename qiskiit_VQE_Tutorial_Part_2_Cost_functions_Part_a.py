### Cost functions

# cost functions describe the goal of the problem, and are also referred to as
# residuals.
# Let's consider a simple example of finding the ground state of a system. Our objective
# is to minimize teh expectation value of the observable rep. energy (hamiltonian)
# min_{\vec{\theta}} <\psi(\vec{\theta}|\hat{H}|\psi(\vec{\theta})>
# we can use the estimator to evaluate the expectation value, and pass the value to an
# optimizer to minimize. If the optimization is successful, it will return a set of optimal
# parameter values \vec{\theta}^* from which we will be able to construct the proposed solution
# state |\psi(\vec{\theta}^*> and compute the observed expectation value as C(\vec{\theta}^*).

# Notice, we will only be able to minimze the cost function for the limited set of states
# that we are considering. This leads us to
#### Our ansatz does not define the solution state across the serach plane: i.e. our optimizer
# will never find the solution, and we need to experiment with out ansatzes that might be able
# to represent our search space more accurately.

#### Our optimizer is unable to find this valid solution: i.e. optimization can be globally
# or locally defined. We want to find the global minimum.

# We will be doing a classical optimatization loop, with evaluation of the cost function on
# a quantum computer.
import numpy as np
import matplotlib.pyplot as plt

def const_func_vqe(params,circuit,hamiltonian,estimator):
    """ Return estimate of energy from estimator

    Parameters:
        params(ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    pub =(circuit,hamiltonian,params)
    cost=estimator.run([pub]).result()[0].data.evs
    return cost

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal

observable = SparsePauliOp.from_list([("XX",1),("YY",-3)])

reference_circuit=QuantumCircuit(2)
reference_circuit.x(0)

variation_form=TwoLocal(
    2,
    rotation_blocks=["rz","ry"],
    entanglement_blocks="cx",
    entanglement="linear",
    reps=1,
)
ansatz=reference_circuit.compose(variation_form)

theta_list=(2*np.pi*np.random.rand(1,8)).tolist()
ansatz.decompose().draw("mpl")
plt.close()

# First, we will use a simulator to carry this out, better to debug locally.
# increasingly problems of interest are no longer classically simulable without state of the art
# supercomputing facilities
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.result import QuasiDistribution
from qiskit.circuit.library import IGate, ZGate, XGate, YGate




estimator=StatevectorEstimator()
cost=const_func_vqe(theta_list,ansatz,observable,estimator)
print(cost)

######## Now quickly make sure you have setup your link to ibm_quantum correctly
####### before this next step.

# we will run on an actual quantum computer!!

#Estimated usage: <1 min:
# Make sure to load the necessary packages
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorOptions, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# Select the least busy backend.

service=QiskitRuntimeService(channel="ibm_quantum")
backend=service.least_busy(
    operational=True, min_num_qubits=ansatz.num_qubits, simulator=False
)
# or get a specfic backend:
#backend=service.backend("ibm_cusco")

# Use the pass manager to traspile the circuit and observable for the specific backend.

pm= generate_preset_pass_manager(backend=backend,optimization_level=1)
isa_ansatz=pm.run(ansatz)
isa_observable=observable.apply_layout(layout=isa_ansatz.layout)

# set estimator options
estimator_options= EstimatorOptions(
    resilience_level=1,
    default_shots=10_000
)

#open a Runtime session:
with Session(backend=backend) as session:
    estimator=Estimator(mode=session, options=estimator_options)
    cost=const_func_vqe(theta_list,isa_ansatz,isa_observable,estimator)

session.close()
print(cost)

############# SUCCESS!!!!!