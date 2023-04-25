from compilers import *
from utils import *
from lattice_hamiltonians import *
from openfermion_hamiltonians import *
import json
from joblib import Parallel, delayed
from datetime import datetime

if __name__ == "__main__":
    cluster = True 
    in_parallel = True
    optimized_run = False
    job_name = 'heisenberg_'
    n_proc = 40

    hamiltonian = heisenberg_hamiltonian(length = 8, b_field = 5, rng_seed=1, b_rand=False)
    hamiltonian = normalize_hamiltonian(hamiltonian)
    norm = np.linalg.norm(np.sum(hamiltonian, axis=0), ord=2)

    compopt = CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    compheur113 = CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=3, state_rand=True, exact_qd=True, use_density_matrices=True)
    compheur123 = CompositeSim(hamiltonian, inner_order=1, outer_order=2, nb=3, state_rand=True, exact_qd=True, use_density_matrices=True)
    trotter1 = CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    trotter2 = CompositeSim(hamiltonian, inner_order=2, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    qdrift = CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)

    simulators = [compheur113, compheur123, trotter1, trotter2, qdrift]
    simulators_names = ['compheur113', 'compheur123', 'trotter1', 'trotter2', 'qdrift'] #naming convention numbered is inner outer nb

    opt_sims = [compopt]
    opt_sims_names = ['compopt']
    
    t_i = 0.01 / norm
    t_f= 3/2 * math.pi / norm
    t_steps = 40
    times = list(np.geomspace(t_i, t_f, t_steps))
    epsilon=0.0001

    data = {}
    data["time"] = times
    data["norm"] = norm
    data["epsilon"] = epsilon

    partition_sim(trotter1, "trotter")
    partition_sim(trotter2, "trotter")
    partition_sim(qdrift, "qdrift")
    partition_sim(compheur113, "chop", chop_threshold = 0.75)
    partition_sim(compheur123, "chop", chop_threshold = 0.75)

    if in_parallel == False:
        for i in range(len(simulators_names)):
            if simulators[i].nb != 1:
                set_nb = simulators[i].nb
            else: set_nb = 1
            def cost_t(times):
                return int(exact_cost(simulator=simulators[i], time=times, nb=set_nb, epsilon=epsilon))
            results = []
            if simulators_names[i] != 'qdrift':
                for t in times:
                    results.append(cost_t(t))
                    print('at time ' + str(times.index(t) + 1) + ' for sim ' + str(i+1) + ' out of ' + str(len(simulators_names)))
            else:
                for t in times[:14]:
                    results.append(cost_t(t))
                    print('at time ' + str(times.index(t) + 1) + ' for sim ' + str(i+1) + ' out of ' + str(len(simulators_names)))
            data[simulators_names[i]] = results

    #parallelization of non-optimized routines
    else:
        for i in range(len(simulators)):
            if simulators[i].nb != 1:
                set_nb = simulators[i].nb
            else: set_nb = 1
            def cost_t(times):
                print('here!')
                return exact_cost(simulator=simulators[i], time=times, nb=set_nb, epsilon=epsilon)
            if simulators_names[i] != 'qdrift': sim_time = times
            else: sim_time = times[:14]
            results = Parallel(n_jobs=n_proc)(delayed(cost_t)(t) for t in sim_time)
            print('done first!')
            data[simulators_names[i]] = results #this requires going back and checking the order of the simulation list, I dont know of a robust workaround at the moment 

        
        if optimized_run == True:
            for j in range(len(opt_sims)):
                def opt_cost_t(times):
                    partition_sim(opt_sims[j], "exact_optimal_chop", time=times, epsilon=epsilon)
                    return int(opt_sims[j].gate_count)
                
                results = Parallel(n_jobs=n_proc)(delayed(opt_cost_t)(t) for t in sim_time)
                data[opt_sims_names[j]] = results #this requires going back and checking the order of the simulation list, I dont know of a robust workaround at the moment 

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M")
        
    if cluster == False:
        filepath = os.path.join(os.path.expanduser('~'), 'Desktop', 'sim_runs', job_name + dt_string)
    else:
        filepath = os.path.join(os.environ['SCRATCH'], job_name + dt_string)
    # open the file for writing and specify the file path
    with open(filepath, 'x') as f:
        # use json.dump to write the dictionary to the file
        json.dump(data, f)
