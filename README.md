# Trotter-Qdrift-Simulation

This library contains code necessary for analyzing the performance of algorithms for the paper [Composite QDrift-Product Formulas for Quantum and Classical Simulations in Real and Imaginary Time](https://arxiv.org). The main goal of the code in this repository is to generate hamiltonians, study the exact performance of composite quantum simulation algorithms (for real time, imaginary time, and Lieb-Robinson localized Hamiltonians), and output necessary plots. 

## Hamiltonian Generation

The Hamiltonians studied here consist of small electronic structure Hamiltonians, such as Hydrogen chains and Lithium-Hydrogen, the Uniform Electron Gas (Jellium), and spin graphs. If one wants to generate the electronic structure Hamiltonians, PySCF and openfermion packages will need to be installed. For Jellium, openfermion will be needed. The spin models are all generated with no external dependencies. These can be accessed in the files `openfermion_hamiltonians.py` and `lattice_hamiltonians.py`. Example ways of generating these can be seen in `PySCFMolecules.ipynb`.

## Algorithm Performance

The bulk of the logic for implementing the simulations is contained in `compilers.py`. This has code to generate Trotter, QDrift, Composite, and Lieb-Robinson (LRSim) local simulators. The main purpose of these objects is to simulate the Hamiltonian provided at construction for a specified amount of time and output the final state. The Composite and LRSim objects can be specified with partitions to determine which terms are randomly sampled and which are performed deterministically. The simulators keep track of how many matrix exponentials of the form e^{i H_i t} are applied to give an accurate idea of the gate cost needed to perform the simulation. The functions used to determine the optimal iterations, number of samples, and partitionings can be found in `utils.py`. Below is an example of simulating a 1D Ising model for 0.5 seconds using a Composite simulation with a "chop" partition.

### Example
```
from lattice_hamiltonians import *
from compilers import CompositeSim
from utils import *

ising_hamiltonian = ising_model(10, b_field = 1.0)
comp_sim = CompositeSim(ising_hamiltonian, nb=20)
partition_sim(comp_sim, "chop", chop_threshold=1.0)
time = 0.5
iterations = 10
# Gets the output of the simulation of a random initial basis state.
comp_sim.simulate(time, iterations)
```

## Data Collection

The simulators are interacted with mostly through the Jupyter notebooks `Hybridized Simulation Jellium.ipynb`, `QMC.ipynb`, as well as `cluster.py` for running parallelized simulations on a compute cluster. 
