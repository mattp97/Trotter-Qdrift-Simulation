from utils import *
from compilers import *
import numpy as np

import pickle
import os

INFIDELITY_TEST_TYPE = "infidelity"
TRACE_DIST_TEST_TYPE = "trace_distance"
GATE_COST_TEST_TYPE = "gate_cost"
CROSSOVER_TEST_TYPE = "crossover"

def cluster_entry_point(directory_path):
    if directory_path[-1] != "/":
        directory_path += "/"
    if os.path.exists(directory_path + "hamiltonian.pickle") == False:
        return
    if os.path.exists(directory_path + "settings.pickle") == False:
        return
    exp = Experiment(output_directory=directory_path)
    exp.load_hamiltonian()
    exp.load_settings()
    exp.run()
    exp.pickle_results()


# For coordinating runs. 
class Experiment:
    def __init__(
        self,
        hamiltonian_list = [],
        output_directory="./",
        experiment_label="default",
        t_start=1e-3,
        t_stop=1e-1,
        t_steps=50,
        num_state_samples=5,
        partitions=["first_order_trotter", "qdrift"],
        test_type=INFIDELITY_TEST_TYPE,
        infidelity_threshold=0.05,
        use_density_matrices=False,
        verbose=False
        ):
        self.sim = CompositeSim(hamiltonian_list=hamiltonian_list, use_density_matrices=use_density_matrices)
        self.use_density_matrices = use_density_matrices
        self.times = np.geomspace(t_start, t_stop, t_steps)
        self.partitions = partitions
        self.test_type=test_type.lower()
        self.infidelity_threshold = infidelity_threshold
        self.verbose = verbose
        self.num_state_samples = num_state_samples
        self.output_directory = output_directory
        self.experiment_label = experiment_label
    
    def pickle_hamiltonian(self):
        ham_list = self.sim.get_hamiltonian_list()
        shape = ham_list[0].shape
        # convert to pickle'able datatype
        to_pickle = [mat.tolist() for mat in ham_list]
        to_pickle.append(shape)
        pickle(to_pickle, open(self.output_directory + "hamiltonian.pickle", 'wb'))
    
    def load_hamiltonian(self):
        unpickled = pickle.load(open(self.output_directory + "hamiltonian.pickle", 'rb'))
        output_shape = unpickled[-1]
        ham_list = []
        for ix in range(len(output_shape) - 1):
            ham_list.append(np.array(unpickled[ix]).reshape(output_shape))
        self.sim.set_hamiltonian(ham_list)
    
    # TODO: implement multithreading?
    # TODO: How to handle probabilistic partitionings?
    def run(self):
        results = {}
        results["times"] = np.copy(self.times).tolist()
        for partition in self.partitions:
            if self.verbose:
                print("[Experiment.run] evaluating partition type:", partition)
            outputs = []
            partition_sim(self.sim, partition_type=partition)
            heuristic = -1
            for t in results["times"]:
                if self.verbose:
                    print("[Experiment.run] evaluating time:", t)
                out = 0
                for _ in range(self.num_state_samples):
                    self.sim.randomize_initial_state()
                    if self.test_type == "infidelity":
                        inf_temp, _ = single_infidelity_sample(self.sim, t)
                        out += inf_temp
                    elif self.test_type == "gate_cost":
                        cost, iters = find_optimal_cost(self.sim, t, self.infidelity_threshold, heuristic=heuristic, verbose=self.verbose)
                        heuristic = iters
                        out += cost
                outputs.append(out / self.num_state_samples)
            results[partition] = outputs
        self.results = results

    def pickle_results(self):
        pickle.dump(self.results, open(self.output_directory + "results.pickle", 'wb'))

    def pickle_settings(self):
        settings = {}
        settings["experiment_label"] = self.experiment_label
        settings["verbose"] = self.verbose
        settings["use_density_matrices"] = self.use_density_matrices
        settings["t_start"] = self.times[0]
        settings["t_stop"] = self.times[-1]
        settings["t_steps"] = len(self.times)
        settings["partitions"] = self.partitions
        settings["infidelity_threshold"] = self.infidelity_threshold
        settings["num_state_samples"] = self.num_state_samples
        settings["output_directory"] = self.output_directory
        settings["test_type"] = self.test_type
        pickle.dump(settings, open(self.output_directory + "settings.pickle", 'wb'))

    # TODO: Replace dictionary access with get and defaults
    def load_settings(self):
        settings = pickle.load(open(self.output_directory + "settings.pickle", 'rb'))
        self.experiment_label     = settings['experiment_label']
        self.verbose              = settings["verbose"]
        self.use_density_matrices = settings["use_density_matrices"]
        self.times                = np.geomspace(settings["t_start"], settings["t_stop"], settings["t_steps"])
        self.partitions           = settings["partitions"]
        self.infidelity_threshold = settings["infidelity_threshold"]
        self.num_state_samples    = settings["num_state_samples"]
        self.test_type            = settings["test_type"]
        # Drop self.output_directory? if we can find the settings then just output there

def test_qdrift():
    graph_ham = graph_hamiltonian(4, 2, 1) 
    e = Experiment(graph_ham, t_start=1e-5, t_stop=1e-2, t_steps=20, verbose=True, experiment_label="testing class performance")
    e.run()
    e.pickle_results()

if __name__ == "__main__":
    test_qdrift()

