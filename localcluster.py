from compilers import *
from utils import *
from lattice_hamiltonians import *
import json
from joblib import Parallel, delayed
from datetime import datetime
from itertools import product

if __name__ == "__main__":
    #sim params
    cluster = True
    n_proc = 40
    job_name = 'local_heisenberg_9'
    coupling = 0.0005
    h_params = local_heisenberg_hamiltonian(length=9, coupling=coupling)
    hamiltonian = (normalize_hamiltonian(h_params[0]), h_params[1], h_params[2])
    norm = np.linalg.norm(np.sum(hamiltonian[0], axis=0), ord=2)
    print("local model shape: " + str(hamiltonian[0].shape))

    #time evolution
    t_i = 0.1 / norm
    t_f= 1/norm #(3/2) * math.pi /norm
    t_steps = 2
    times = list(np.geomspace(t_i, t_f, t_steps))
    blocks = list(range(2, 5, 1))
    print(blocks)
    block_nb = [3, 3, 3]
    epsilon=0.00001

    partitions = ["optimal_chop", "trotter", "chop"]
    
    data = {}
    data['partitions'] = partitions
    data["time"] = times
    data["norm"] = norm
    data["epsilon"] = epsilon
    data["blocks"] = blocks
    data["block_nb"] = block_nb
    data["ham_dims"] = hamiltonian[0].shape
    data['coupling'] = coupling

    sim_params = list(product(blocks, partitions, times))
    print(sim_params)
        
    def cost_t(params):
        blocked_hamiltonian = hamiltonian_localizer_1d(hamiltonian, sub_block_size=params[0])
        local_sim = LRsim(hamiltonian[0], blocked_hamiltonian, inner_order=1, nb=block_nb)
        if params[1] != "optimal_chop":
            local_partition(simulator=local_sim, partition=params[1], weights = [0.1, 0.1, 0.1])
            cost =  exact_cost(local_sim, time=params[2], nb=block_nb, epsilon=epsilon)
        else:
            local_partition(simulator=local_sim, partition=params[1])
            cost = local_sim.gate_count
        return cost
    
    results = Parallel(n_jobs=n_proc)(delayed(cost_t)(p) for p in sim_params)

    array_3d = (np.array(results).reshape(len(blocks), len(partitions), len(times))).astype(float)

    for i in range(len(blocks)):
        for j in range(len(partitions)):
            data[partitions[j] + "_block_"+str(blocks[i])] = list(array_3d[i,j,:])

    # #unpack results -- this will have to be modified if we include more partitions (one if statement should do it)
    # n = len(times) # number of elements to group together
    # count = 0
    # index = -1
    # b = len(blocks)
    # for j in results:
    #     if count % n == 0: 
    #         index +=1 
    #         # create a new list for every n elements
    #         current_list = []
    #         # associate the new list with a dictionary key
    #         data['block ' + str(blocks[index])] = current_list
    #     # add the current element to the current list
    #     current_list.append(j)
    #     count += 1
    
    def trot_cost(t):
        sim = CompositeSim(hamiltonian[0], state_rand=True, use_density_matrices=True, exact_qd=True)
        partition_sim(sim, 'trotter')
        return exact_cost(sim, t, nb=1, epsilon=epsilon)
    
    def qd_cost(t):
        sim = CompositeSim(hamiltonian[0], state_rand=True, use_density_matrices=True, exact_qd=True)
        partition_sim(sim, 'qdrift')
        return exact_cost(sim, t, nb=1, epsilon=epsilon)
    
    trot_results = Parallel(n_jobs=n_proc)(delayed(trot_cost)(t) for t in times)
    qd_results = Parallel(n_jobs=n_proc)(delayed(qd_cost)(t) for t in times[:(int(3/4 * len(times)))])
    
    data['trotter'] = trot_results
    data['qdrift'] = qd_results
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