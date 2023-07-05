# Trotter-Qdrift-Simulation

This library contains code necessary for analyzing the performance of algorithms for the paper [Composite QDrift-Product Formulas for Quantum and Classical Simulations in Real and Imaginary Time](https://arxiv.org/abs/2306.16572). The main goal of the code in this repository is to generate Hamiltonians, study the exact performance of composite quantum simulation algorithms (for real-time, imaginary-time, and Lieb-Robinson localized Hamiltonians), and output necessary plots. 

## Hamiltonian Generation

The Hamiltonians studied here consist of small electronic structure Hamiltonians, such as Hydrogen chains and Lithium-Hydrogen, the Uniform Electron Gas (Jellium), and spin graphs. If one wants to generate the electronic structure Hamiltonians, PySCF and OpenFermion packages will need to be installed (https://github.com/quantumlib/OpenFermion-PySCF). For Jellium, OpenFermion will be needed. The spin models are all generated with no external dependencies. These can be accessed in the files `openfermion_hamiltonians.py` and `lattice_hamiltonians.py`. Example ways of generating these can be seen in `PySCFMolecules.ipynb`.

## Algorithm Performance

The bulk of the logic for implementing the simulations is contained in `compilers.py`. This has code to generate Trotter, QDrift, and Composite simulators. Lieb-Robinson (LRSim) local simulators are contained in `localcluster.py`. The main purpose of these objects is to simulate the Hamiltonian provided at construction for a specified amount of time and output the final state. The Composite and LRSim objects can be specified with partitions to determine which terms are randomly sampled and which are performed deterministically. The simulators also provide the means for building the exact density matrices (as was done in the paper linked above), or to sample the propagators and work with state vector formalism. The simulators keep track of how many matrix exponentials of the form exp(-i H_i t) are applied to give an accurate idea of the gate cost needed to perform the simulation. The functions used to determine the optimal iterations, number of samples, and partitionings can be found in `utils.py`. Below is an example of simulating a 1D Ising model for 0.5 seconds using a Composite simulation with a "chop" partition.

### Example
```
from lattice_hamiltonians import *
from compilers import CompositeSim
from utils import *

ising_hamiltonian = ising_model(8, b_field = 5.0)
comp_sim = CompositeSim(ising_hamiltonian, nb=4, state_rand=True) #nb = qdrift samples
partition_sim(comp_sim, "chop", chop_threshold=1.0)
time = 0.5
iterations = 20
# Gets the output of the simulation of a random initial basis state.
final_state = comp_sim.simulate(time, iterations)
```
We can then use functions to find the error in this simulator, or we can calculate the total gate count this simulator will require to meet our desired precision at this time.

```
#epsilon error 
epsilon = sim_trace_distance(comp_sim, time=0.5, iterations=20, nb=4)

#gate count for a chosen precision of 0.001
cost = exact_cost(simulator=comp_sim, time=0.5, nb=4, epsilon=0.001)
```

If we wish to optimize simulation parameters, choosing the partition `exact_optimal_chop` will automatically find the optimal gate count in calculating the best partitions and N_B values. This gate count can then be accessed through the method `comp_sim.gate_count`. 

## Data Collection

The simulators are interacted with mostly through the Jupyter notebook `QMC.ipynb`, as well as `cluster.py` for running parallelized simulations on a compute cluster. 
