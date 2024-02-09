# Trotter-Qdrift-Simulation

This library contains code necessary for analyzing the performance of algorithms for the paper [Composite QDrift-Product Formulas for Quantum and Classical Simulations in Real and Imaginary Time](https://arxiv.org/abs/2306.16572). The main goal of the code in this repository is to generate hamiltonians, study the exact performance of composite quantum simulation algorithms (for real time, imaginary time, and Lieb-Robinson localized Hamiltonians), and output necessary plots. This is an important task in quantum algorithms given that calculating precise constant factors in the scaling for simulation of specific Hamiltonians can be incredibly tedious and challenging, and upper bounds often do not tell the full story.

## Hamiltonian Generation

The Hamiltonians studied here consist of small electronic structure Hamiltonians, such as Hydrogen chains and Lithium-Hydrogen, the Uniform Electron Gas (Jellium), and spin models on lattices/graphs. If one wants to generate the electronic structure Hamiltonians, PySCF and openfermion packages will need to be installed. For Jellium, openfermion will be needed. The spin models are all generated with no external dependencies. These can be accessed in the files `openfermion_hamiltonians.py` and `lattice_hamiltonians.py`. Example ways of generating these can be seen in `PySCFMolecules.ipynb`.

## Algorithm Performance

The bulk of the logic for implementing the simulations is contained in `compilers.py`. This has code to generate Trotter, QDrift, and Composite simulators. Lieb-Robinson (LRSim) local simulators are containted in `localcluster.py`. The main purpose of these objects is to simulate the Hamiltonian provided at construction for a specified amount of time and output the final state. The Composite and LRSim objects can be specified with partitions to determine which terms are randomly sampled and which are performed deterministically. The simulators also provide the means for building the exact density matrices (as was done in the paper linked above), or to sample the propagators and work with state vector formalism. The simulators keep track of how many matrix exponentials of the form ![equation](https://latex.codecogs.com/svg.image?&space;e^{-iH_j&space;t/r}) are applied to give an accurate idea of the gate cost needed to perform the simulation. The functions used to determine the optimal iterations, number of samples, and partitionings can be found in `utils.py`. Below is an example of simulating a 1D Heisenberg Model for 0.5 seconds using a Composite simulation with a "chop" partition. Since the error of product formulas becomes periodic, it is recommended to normalize the Hamiltonian and simulate for time t< 3/2pi.

### Detailed Example
```
from lattice_hamiltonians import *
from compilers import CompositeSim
from utils import *

heisenberg_hamiltonian = heisenberg_model(length=8, b_field = 5.0) #length sets number of sites and b_field sets the strength of the magnetic field (coupling parameter is set to 1)
```
As discussed above, we should normalize the Hamiltonian. This library considers 2 kinds of normalization: one in which the spectral norm of the maximum Hamiltonian summand divides all Hamiltonian terms (this is helpful in later choosing a chop parameter), and one in which ||H|| = 1 where H is the full Hamiltonian (note these are not the same by the triangle inequality). Below we both normalize H in the first sense by calling `normalize_hamiltonian`, and then we extract the total norm to divide the time later. 
```
normed_hamiltonian = normalize_hamiltonian(heisenberg_hamiltonian)
norm = np.linalg.norm(np.sum(normed_hamiltonian, axis=0), ord=2)
comp_sim = CompositeSim(normed_hamiltonian, nb=20, state_rand=True, exact_qd=True, use_density_matrices=True, imag_time=False) #sets the qdrift samples to 20 and randomizes the initial state
```
CompositeSim will take any 3d numpy array as a Hamiltonian input. When performing calculations regarding quantum simulation and product formula cost analysis, we have freedom in choosing between the density matrix or state vector formalism. This library and CompositeSim objects permit for the use of both via the `use_density_matrices` flag. However, if we use density matrices, then it is possible to build the exact QDrift channel produced by the simulation (as opposed to sampling via RNG) via the `exact_qd` flag. The convinience of this is that it guarentees monotonicity of the error, which allows for simplified cost analysis. Doing this with state vectors is not possible, given the inability to write down mixed states. While both of these formalisms work in the `.simulate` method, it is strongly recommended to use density matrices and exact Qdrift if one wishes to use functions regarding optimization and cost analysis, as all of these functions were written with the trace distance and monotonicity in mind. Also, note that if you wish to work in imaginary time, simply set `imag_time=True`. Never give `.simulate` a complex time. `CompositeSim` also has the method `set_initial_state()` to give the simulator the user's state of choice (see function docstrings). Next, we partition the simulator and run the simulation.
```
partition_sim(comp_sim, "chop", chop_threshold=0.25) #runs the chop partition from the paper. This places each Hamiltonian term with norm < 0.25 in Qdrift
time = 0.5/norm #from above
iterations = 20
# Gets the output of the simulation of a random initial basis state.
final_state = comp_sim.simulate(time, iterations) #the simulate method is what runs the simulation (applies the product formula). final_state is a density matrix

gates_applied = comp_sim.gate_count #the CompositeSim object stores the gates applied in the previous simulation as an attribute
```

We can then use functions to find the error in this simulator, or we can calculate the total gate count this simulator will require to meet our desired precision at this time.
```
#epsilon error 
epsilon = sim_trace_distance(comp_sim, time=0.5, iterations=20, nb=4)

#gate count for a chosen precision of 0.001
cost = exact_cost(simulator=comp_sim, time=0.5, nb=4, epsilon=0.001)
```
Note that calculating `epsilon` in this manner runs the simulation again, so one can also extract the `gate_count` after running `sim_trace-distance`. Epsilon can also be calculated using an arbitrary distance measure by using the state returned by `.simulate`. The file `utils.py` contains both the `trace_distance()` and `infidelity()` built in. To use these functions to compare `final_state` with that given by the exact simulation, one can call `exact_time_evolution_density()` when time is real or `exact_imaginary_channel()` otherwise. The `exact_cost` function solves the search problem of finding the minimum number of iterations to achieve error epsilon by using both exponential and binary search (by brute-force simulating and checking). For large Hamiltonians and time close to 2pi, this can become quite expensive. 

If we wish to optimize simulation parameters, choosing the partition `exact_optimal_chop` will automatically find the optimal gate count in calculating the best partitions and N_B values. Different from other partition methods, one does not need to partition the simulator and then run `.simulate` or `exact_cost()` to extract `.gate_count`. Rather `exact_optimal_chop` finds the minimum gate count in the process of optimizing the partition. This gate count can then be accessed in the usual way. An example is below. 
```
epsilon = 0.001 #set error tolerance
t= 0.5/norm
partition_sim(comp_sim, "exact_optimal_chop", time=t, epsilon=epsilon, q_tile=90)
optimal_params = comp_sim.gate_count
```
Within `partition_sim()` the `q_tile` variable calculates the given quantile, or term corresponding to said percentile, from the spctral norm distribution of the Hamiltonian terms (already contained in the `comp_sim` object) and sets this as an upper bound for the optimization over the chop parameter/threshold. This is done purely to optimize the machine learning tree algorithm that runs under the hood. The intuition is that a chop parameter that falls in a very large percentile of the terms will inevitably place some very large terms in the QDrift channel which will make the simulation expensive to run. Given that the optimization works by repeated calls to `exact_cost()`, avoiding searching the high q_tile space leads to much more efficient runs. However, if one does not want this upper bound in the search space, set `q_tile=100`. Upon completion, `optimal_params` will store the optimized gate count, N_B, and chop threshold like so ` (gate_count, [N_B, chop_threshold])`.


## Data Collection
The simulators are interacted with mostly through the Jupyter notebook `QMC.ipynb`, as well as `cluster.py` for running parallelized simulations on a compute cluster. There are no specific structures to collect results for repeated experiments as these are often very use-case dependent. 
