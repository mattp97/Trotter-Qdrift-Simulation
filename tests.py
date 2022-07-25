from utils import *
from compilers import *
import numpy as np
import matplotlib.pyplot as plt
import time as timeit

# For coordinating runs. 
class Experiment:
    def __init__(self, hamiltonian_list):
        sim = CompositeSim(hamiltonian_list=hamiltonian_list)        

# Compares a composite simulator with the given partition types across the specified
# time range. Returns a result dictionary that maps "times" to the array of times 
# the simulators were evaluated and maps each partition string to the specified output
# from "test_type" which defaults to infidelity. 
def evaluate_simulator(
        simulator,
        t_start=1e-3,
        t_end=1e-1,
        t_steps=50,
        num_state_samples=5,
        partitions=["first_order_trotter", "qdrift"],
        test_type="infidelity",
        infidelity_threshold=0.05,
        verbose=False
    ):
    times = np.geomspace(t_start, t_end, t_steps)
    results = dict()
    for partition in partitions:
        vals = []
        partition_sim(simulator, partition_type=partition)
        simulator.print_partition()
        heuristic = -1
        print("[testing] on partition:", partition)
        count = 0
        for t in times:
            if count % 10 == 0:
                print("[testing] time:", t)
            count += 1
            val = 0
            for _ in range(num_state_samples):
                simulator.randomize_initial_state()
                if test_type == "infidelity":
                    inf_temp, _ = single_infidelity_sample(simulator, t)
                    val += inf_temp
                elif test_type == "gate_cost":
                    cost, iters = find_optimal_cost(simulator, t, infidelity_threshold, heuristic=heuristic, verbose=verbose)
                    heuristic = iters
                    val += cost

            vals.append(val / num_state_samples)
        results[partition] = vals
    results["times"] = times
    return results

def test_qdrift():
    graph_ham = graph_hamiltonian(4, 2, 1)
    print("[qdrift] hamiltonian:")
    print(len(graph_ham))
    sim = CompositeSim(graph_ham)
    partition_sim(sim, "qdrift")
    # From results of a previous run, it looks like nb=225 for graph_hamiltonian(4, 2, 1) yields reasonable infidelity. Going to
    # Optimize QDrift performance at this metric.  Unoptimized it took 21.259 for the cumtime results from profile. 
    multi_infidelity_sample(sim, 0.1, nbsamples=225, mc_samples=100)

if __name__ == "__main__":
    test_qdrift()
