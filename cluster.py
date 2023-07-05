from compilers import *
from utils import *
from lattice_hamiltonians import *
from openfermion_hamiltonians import *
import json
from joblib import Parallel, delayed
from datetime import datetime
from itertools import product

if __name__ == "__main__":
    cluster = False
    job_name = 'LiH_'
    n_proc = 12
    imag_flag = False

    #hamiltonian = exp_loc_graph_hamiltonian(7, 1, 1)
    #hamiltonian = exp_distr_heisenberg_hamiltonian(length = 7, b_field = 5, rng_seed=1, b_rand=False)
    #hamiltonian = jellium_hamiltonian(dimensions=1, length=4, spinless=True)
    #hamiltonian = hydrogen_chain_hamiltonian(chain_length=2, bond_length=0.8)
    hamiltonian = LiH_hamiltonian()
    hamiltonian = normalize_hamiltonian(hamiltonian)
    norm = np.linalg.norm(np.sum(hamiltonian, axis=0), ord=2)

    #store the simulator then its name
    compopt = [CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True, imag_time=imag_flag), 'compopt']
    compopt12 = [CompositeSim(hamiltonian, inner_order=1, outer_order=2, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True , imag_time=imag_flag), 'compopt12']
    compopt21 = [CompositeSim(hamiltonian, inner_order=2, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True , imag_time=imag_flag), 'compopt21']
    compopt22 = [CompositeSim(hamiltonian, inner_order=2, outer_order=2, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True , imag_time=imag_flag), 'compopt22']
    #compheur11_2 = [CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=2, state_rand=True, exact_qd=True, use_density_matrices=True, imag_time=imag_flag), 'compheur11_2']
    #compheur12_4 = [CompositeSim(hamiltonian, inner_order=1, outer_order=2, nb=4, state_rand=True, exact_qd=True, use_density_matrices=True, imag_time=imag_flag), 'compheur12_4']
    trotter1 = [CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True, imag_time=imag_flag), 'trotter1']
    trotter2 = [CompositeSim(hamiltonian, inner_order=2, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True, imag_time=imag_flag), 'trotter2']
    qdrift = [CompositeSim(hamiltonian, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True, imag_time=imag_flag), 'qdrift']

    def_sims = [trotter1, trotter2, qdrift] #definite sims
    opt_sims = [compopt, compopt12, compopt21, compopt22] #sims with optimized parameters
    simulators = opt_sims + def_sims

    opt_names = [sublist[1] for sublist in opt_sims]
    
    t_i = 0.01 / norm
    t_f = (3/2) * math.pi / norm
    t_steps = 20
    times = list(np.geomspace(t_i, t_f, t_steps))
    epsilon=0.001

    sim_params = list(product(simulators, times))
    qd_times = times[:int((3/4) * len(times))]

    data = {}
    data["time"] = times
    data["ham_dims"] = hamiltonian.shape
    print(hamiltonian.shape)
    data["is_time_imaginary"] = imag_flag
    data["qdtime"] = qd_times
    data["norm"] = norm
    data["epsilon"] = epsilon

    partition_sim(trotter1[0], "trotter")
    partition_sim(trotter2[0], "trotter")
    partition_sim(qdrift[0], "qdrift")
    #partition_sim(compheur113, "chop", chop_threshold = 0.75)
    #partition_sim(compheur11_2[0], "chop", chop_threshold = 0.75)
    #partition_sim(compheur12_4[0], "chop", chop_threshold = 0.75)

    def cost_t(params):
        if params[0][0].nb != 1:
             set_nb = params[0][0].nb
        else: set_nb = 1
        if params[0][1] in opt_names:
            partition_sim(params[0][0], "exact_optimal_chop", time=params[1], epsilon=epsilon, q_tile=90)
            cost = params[0][0].gate_count
        else:
            if (params[0][1] == 'qdrift') and (params[1] in qd_times):
                cost = exact_cost(simulator=params[0][0], time=params[1], nb=set_nb, epsilon=epsilon)

            elif (params[0][1] == 'qdrift') and (params[1] not in qd_times):
                cost = None

            else: 
                cost = exact_cost(simulator=params[0][0], time=params[1], nb=set_nb, epsilon=epsilon)
        return cost 

    results = Parallel(n_jobs=n_proc)(delayed(cost_t)(p) for p in sim_params)

    #unpack results -- this will have to be modified if we include more partitions (one if statement should do it)
    n = len(times) # number of elements to group together
    count = 0
    sim_index = -1
    for j in results:
        if count % n == 0:
            sim_index+=1
            # create a new list for every n elements
            current_list = []
            # associate the new list with a dictionary key
            print(sim_index)
            data[simulators[sim_index][1]] = current_list
        # add the current element to the current list
        current_list.append(j)
        count += 1
    print(data)

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
    
if cluster == False:
    filepath = os.path.join(os.path.expanduser('~'), 'Desktop', 'sim_runs', job_name + dt_string)
else:
    filepath = os.path.join(os.path.expanduser('~'), 'compsim_data', job_name + dt_string)
# open the file for writing and specify the file path
with open(filepath, 'x') as f:
    # use json.dump to write the dictionary to the file
    json.dump(data, f)
