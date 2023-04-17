from compilers import *
from utils import *
from lattice_hamiltonians import *
import json
import multiprocessing as mp
from datetime import datetime

if __name__ == "__main__":
    optimized_run = False
    hamiltonian = heisenberg_hamiltonian(length = 8, b_field = 5, rng_seed=1, b_rand=False)
    hamiltonian = normalize_hamiltonian(hamiltonian)
    norm = np.linalg.norm(np.sum(hamiltonian, axis=0), ord=2)

    compopt = CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    compheur11 = CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=4, state_rand=True, exact_qd=True, use_density_matrices=True)
    compheur12 = CompositeSim(hamiltonian, inner_order=1, outer_order=2, nb=4, state_rand=True, exact_qd=True, use_density_matrices=True)
    trotter1 = CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    trotter2 = CompositeSim(hamiltonian, inner_order=2, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    qdrift = CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)

    simulators = [compheur11, compheur12, trotter1, trotter2, qdrift]
    simulators_names = ['compheur11', 'compheur12', 'trotter1', 'trotter2', 'qdrift']

    opt_sims = [compopt]
    opt_sims_names = ['compopt']
    

    t_i = 0.01
    t_f= 3/2 * math.pi / norm
    t_steps = 40
    times = np.geomspace(t_i, t_f, t_steps)
    epsilon=0.0001

    data = {}
    data["time"] = times
    data["norm"] = norm
    data["epsilon"] = epsilon

    partition_sim(trotter1, "trotter")
    partition_sim(trotter2, "trotter")
    partition_sim(qdrift, "qdrift")
    partition_sim(compheur11, "chop", chop_threshold = 0.75)
    partition_sim(compheur12, "chop", chop_threshold = 0.75)

    #parallelization of non-optimized routines
    for i in range(len(simulators)):
        if simulators[i].nb != 1:
            set_nb = simulators[i].nb
        else: set_nb = 1
        def cost_t(times):
            return exact_cost(simulator=simulators[i], time=times, nb=set_nb, epsilon=epsilon)
        with mp.Pool(processes=4) as pool: #set processes to 40 for the cluster
            results_iterator = pool.imap(cost_t, times)
            results = [result for result in results_iterator]
        data[simulators_names[i]] = results #this requires going back and checking the order of the simulation list, I dont know of a robust workaround at the moment 

    if optimized_run == True:
        for j in range(len(opt_sims)):
            def opt_cost_t(times):
                partition_sim(opt_sims[j], "exact_optimal_chop", time=times, epsilon=epsilon)
                return int(opt_sims[j].gate_count)
            with mp.Pool(processes=4) as pool: #set processes to 40 for the cluster
                results_iterator = pool.imap(opt_cost_t, times)
                results = [result for result in results_iterator]
            data[opt_sims_names[j]] = results #this requires going back and checking the order of the simulation list, I dont know of a robust workaround at the moment 

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y_%H:%M")
        
    filepath = '$SCRATCH/compsim_data/' + dt_string
    # open the file for writing and specify the file path
    with open(filepath, 'w') as f:
        # use json.dump to write the dictionary to the file
        json.dump(data, f)
