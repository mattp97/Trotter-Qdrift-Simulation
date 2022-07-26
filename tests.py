import sys
from utils import *
from compilers import *
import numpy as np

import pickle
import os

INFIDELITY_TEST_TYPE = "infidelity"
TRACE_DIST_TEST_TYPE = "trace_distance"
GATE_COST_TEST_TYPE = "gate_cost"
CROSSOVER_TEST_TYPE = "crossover"


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
        if output_directory[-1] != '/':
            self.output_directory = output_directory + '/'
        else:
            self.output_directory = output_directory
        self.experiment_label = experiment_label
    
    def pickle_hamiltonian(self, input_path):
        ham_list = self.sim.get_hamiltonian_list()
        shape = ham_list[0].shape
        # convert to pickle'able datatype
        to_pickle = [mat.tolist() for mat in ham_list]
        to_pickle.append(shape)
        pickle(to_pickle, open(input_path, 'wb'))
    
    def load_hamiltonian(self, output_path):
        unpickled = pickle.load(open(output_path, 'rb'))
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
        pickle.dump(self.results, open(self.output_directory + "results.pickle", 'wb'))

    def pickle_settings(self, output_path):
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
        pickle.dump(settings, open(output_path, 'wb'))

    # TODO: Replace dictionary access with get and defaults
    def load_settings(self, settings_path):
        settings = pickle.load(open(settings_path, 'rb'))
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

def find_pickles():
    if len(sys.argv) == 3:
        scratch_path = sys.argv[-1]
    else:
        scratch_path = os.getenv("SCRATCH")
        if type(scratch_path) != type("string"):
            print("[find_pickle] Error, could not find SCRATCH environment variable")
            return None, None, None
    if scratch_path[-1] != '/':
        scratch_path += '/'
    if os.path.exists(scratch_path + 'hamiltonian.pickle'):
        ham_path = scratch_path + 'hamiltonian.pickle'
    else:
        ham_path = None
    if os.path.exists(scratch_path + 'settings.pickle'):
        settings_path = scratch_path + 'settings.pickle'
    else:
        settings_path = None
    return ham_path, settings_path, scratch_path
    
def client_entry_point():
    if len(sys.argv) == 3:
        output_dir = sys.argv[-1]
    else:
        print("[client_entry_point] No output directory given, using SCRATCH")
        scratch_path = os.getenv("SCRATCH")
        if type(scratch_path) != type("string"):
            print("[client_entry_point] No directory given and no scratch path")
        output_dir = scratch_path
    if output_dir[-1] != '/':
        output_dir += '/'
    ham_list = graph_hamiltonian(4,2,1)
    shape = ham_list[0].shape
    pickle_ham = [mat.tolist() for mat in ham_list]
    pickle_ham.append(shape)
    experiment_label = input("label for the experiment (string): ")
    verbose = input("verbose (True/False): ")
    use_density_matrices = input("use_density_matrices (True/False): ")
    t_start = input("t_start (float): ")
    t_stop = input("t_stop (float): ")
    t_steps = input("t_steps (positive int): ")
    num_partitions = input("number of partitions: ")
    partitions = []
    for i in range(int(num_partitions)):
        partitions.append(input("enter partition type #" + str(i + 1) +" : "))
    infidelity_threshold = input("infidelity threshold (float): ")
    num_state_samples = input("num_state_samples (positive int): ")
    output_directory = input("output_dir (string): ")
    test_type = input("test_type (string): ")
    settings ={}
    settings["experiment_label"] = experiment_label
    settings["verbose"] = bool(verbose)
    settings["use_density_matrices"] = bool(use_density_matrices)
    settings["t_start"] = float(t_start)
    settings["t_stop"] = float(t_stop)
    settings["t_steps"] = int(t_steps)
    settings["partitions"] = partitions
    settings["infidelity_threshold"] = float(infidelity_threshold)
    settings["num_state_samples"] = int(num_state_samples)
    settings["output_directory"] = output_directory
    settings["test_type"] = test_type
    pickle.dump(settings, open(output_dir + "settings.pickle", 'wb'))
    pickle.dump(pickle_ham, open(output_dir + "hamiltonian.pickle", 'wb'))

def compute_entry_point():
    ham_path, settings_path, scratch_dir = find_pickles()
    if type(ham_path) != type("string") or type(settings_path) != type("string") or type(scratch_dir) != type("string"):
        print("[tests.py] Error: could not find hamiltonian.pickle or settings.pickle")
        sys.exit()
    ham_list = pickle.load(open(ham_path, 'rb'))
    settings = pickle.load(open(settings_path, 'rb'))
    print("[compute_entry_point] hamiltonian found:")
    print(ham_list)
    print("#" * 50)
    print("settings found:")
    print(settings)
    # working_dir = scratch_dir + 'output'
    # os.mkdir(working_dir)
    # exp = Experiment(output_directory=working_dir)
    # exp.load_hamiltonian(ham_path)
    # exp.load_settings(settings_path)
    # exp.run()

if __name__ == "__main__":
    if sys.argv[1] == "client":
        client_entry_point()
    if sys.argv[1] == "compute":
        compute_entry_point()
    

    
