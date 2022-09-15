from compilers import *
from utils import *
import json
import multiprocessing as mp
if __name__ == "__main__":

    heisenberg_hamiltonian_list = heisenberg_hamiltonian(length = 9, b_field=1, rng_seed=1)
    b5_hamiltonian_list = hamiltonian_localizer_1d(heisenberg_hamiltonian_list, sub_block_size=5)
    b3_hamiltonian_list = hamiltonian_localizer_1d(heisenberg_hamiltonian_list, sub_block_size=3)
    normalized_heisenberg = normalize_hamiltonian(heisenberg_hamiltonian_list[0])
    print("local model shape: " + str(heisenberg_hamiltonian_list[0].shape))

    local_b5 = LRsim(heisenberg_hamiltonian_list[0], b5_hamiltonian_list, inner_order=1, nb=[1,1,1], state_rand = False)
    local_b3 = LRsim(heisenberg_hamiltonian_list[0], b3_hamiltonian_list, inner_order=1, nb=[1,1,1], state_rand = False)
    local_trot_b3 =  LRsim(heisenberg_hamiltonian_list[0], b3_hamiltonian_list, inner_order=1, nb=[1,1,1], state_rand = False)
    local_trot_b5 =  LRsim(heisenberg_hamiltonian_list[0], b5_hamiltonian_list, inner_order=1, nb=[1,1,1], state_rand = False)
    trotter1 = CompositeSim(normalized_heisenberg, inner_order=1, outer_order=1, nb=1, state_rand=False, exact_qd=True, use_density_matrices=True)
    qdrift = CompositeSim(normalized_heisenberg, inner_order=1, outer_order=1, nb=1, state_rand=False, exact_qd=True, use_density_matrices=True)
        
    t_i = 0.01
    t_f=1.5
    t_steps = 20
    times = np.geomspace(t_i, t_f, t_steps)
    epsilon=0.001
    
    local_b3_results = dict()
    local_b5_results = dict()
    local_trot_b3_results = dict()
    local_trot_b5_results = dict()
    trotter1_results = dict()
    qdrift_results = dict()
    
    partition_sim(trotter1, "trotter")
    partition_sim(qdrift, "qdrift")
    local_partition(local_trot_b3, "trotter")
    local_partition(local_trot_b5, "trotter")
    
    for t in times:
        print("At time :" +str(t))

        local_partition(local_b3, "optimal_chop", time =t, epsilon=epsilon)
        local_b3_results[t] = int(sim.gate_count)

        local_partition(local_b5, "optimal_chop", time =t, epsilon=epsilon)
        local_b5_results[t] = int(sim.gate_count)

        local_trot_b3_results[t] = int(exact_cost(local_trot_b3, time=t, nb=1, epsilon=epsilon))

        local_trot_b5_results[t] = int(exact_cost(local_trot_b5, time=t, nb=1, epsilon=epsilon))

        trotter1_results[t] = int(exact_cost(trotter1, time=t, nb=1, epsilon=epsilon))

        if t <= times[12]:
            qdrift_results[t] = int(exact_cost(qdrift, time=t, nb=1, epsilon=epsilon))

        if t == times[12]:
            int_out = open("intermediate_local_graph9", "w")
            json.dump(local_b3_results, int_out)
            json.dump(local_b5_results, int_out)
            json.dump(local_trot_b3_results, int_out)
            json.dump(local_trot_b5_results, int_out)
            json.dump(trotter1_results, int_out)
            json.dump(qdrift_results, int_out)
            int_out.close()

    outfile = open("9_local_graph_opchopb3_b5_localtrotb3_b5_trot1_qd", "w")

    json.dump(local_b3_results, outfile)
    json.dump(local_b5_results, outfile)
    json.dump(local_trot_b3_results, outfile)
    json.dump(local_trot_b5_results, outfile)
    json.dump(trotter1_results, outfile)
    json.dump(qdrift_results, outfile)
        
    outfile.close()