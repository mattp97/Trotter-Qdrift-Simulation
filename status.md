# Bugs
- Why is fidelity for density matrices above 1.0 even though simulate seems to preserve trace to very high floating point levels (1e-12) and negativity is observed on level of 2e-5?
- Trace Distance with density matrices also does not seem to be working right.
- Using graph_421 hamiltonian, qdrift only partition, NB = 1, iter_lower = -1, iter_upper=5583, curr_guess=5583 yields a non-terminating get_iteration_bounds loop.
