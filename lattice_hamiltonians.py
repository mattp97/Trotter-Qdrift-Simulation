import numpy as np

# This code includes functions to generate lattice hamiltonians of arbitrary size with desired connectivity
# and other properties

#Globally define some useful operators that will never be changed:
X = np.array([[0, 1],
     [1, 0]])
Z = np.array([[1, 0],
     [0, -1]])
Y = np.array([[0, -1j],
     [1j, 0]])
I = np.array([[1, 0],
     [0, 1]])
XX = np.kron(X, X) #tensor products between two Pauli's
XZ = np.kron(X, Z)
ZZ = np.kron(Z, Z)
ZX = np.kron(Z, X)
II = np.kron(I, I)
IX = np.kron(I, X)
XI = np.kron(X, I)
IZ = np.kron(I, Z)
ZI = np.kron(Z, I)

#Some helper functions for building lattices
# A simple function that computes the graph distance between two sites
def dist(site1, site2):
    distance_vec = site1 - site2
    distance = np.abs(distance_vec[0]) + np.abs(distance_vec[1])
    return distance

# A simple function that initializes a graph in the form of an np.array of coordinates 
def initialize_graph(x_sites, y_sites):
    coord_list = []
    for i in range(x_sites):
        for j in range(y_sites):
            coord_list.append([i,j])
    return np.array(coord_list)

#A funciton that initializes a Pauli operator in the correct space, acting on a specific qubit
def initialize_operator(operator_2d, acting_space, space_dimension):
    if acting_space>space_dimension:
        return 'error'
    for i in range(acting_space):
        operator_2d = np.kron(operator_2d, I)
    for j in range(space_dimension - acting_space-1):
        operator_2d = np.kron(I, operator_2d)
    return operator_2d

def initialize_observable(operator_2d, space_dimension):
    temp_op = np.copy(operator_2d)
    for i in range(space_dimension - 1):
        temp_op = np.kron(temp_op, operator_2d)
    return temp_op 


#Functions that return np arrays (1D lists) containing hamiltonian matrix summands (2D numpy arrays)

#Exponentially decaying local graph hamiltonian. Interactions fall of exponentially with the graph distance
def exp_loc_graph_hamiltonian(x_dim, y_dim, rng_seed):
    np.random.seed(rng_seed)
    hamiltonian_list = []
    graph = initialize_graph(x_dim, y_dim)
    for i in range(x_dim*y_dim):
        for j in range(y_dim*x_dim):
            if i != j: #long range interaction
                alpha = np.random.normal()
                hamiltonian_list.append(alpha * 
                    np.matmul(initialize_operator(Z, i, x_dim*y_dim), initialize_operator(Z, j, x_dim*y_dim)) *
                        10.0**(-dist(graph[i], graph[j]))) 

            # if (dist(graph[i], graph[j])==1) and (i>j): #nearest neighbour interaction
            #     beta = np.random.normal()
            #     hamiltonian_list.append(beta * np.matmul(initialize_operator(Y, i, x_dim*y_dim), initialize_operator(Y, j, x_dim*y_dim)))
            
        gamma = np.random.normal()
        hamiltonian_list.append(4* gamma * initialize_operator(X, i, x_dim*y_dim))
                
    return np.array(hamiltonian_list)


#a function that generates the list of hamiltonian terms for a random NN Heinsenberg model with abritrary b_field strength
# different from the other funciton as this one just returns a hamiltonian (no tracking of indices like in the other .ipynb)
def local_heisenberg_hamiltonian(length, b_field, rng_seed, b_rand):
    #b_rand is a boolean that either sets the field to be randomized or the interactions (if false)
    y_dim = 1
    x_dim = length #restrict to 1d spin change so we can get more disjoint regions
    np.random.seed(rng_seed)
    hamiltonian_list = []
    indices = []
    #graph = initialize_graph(x_dim, y_dim)
    operator_set = [X, Y, Z]
    lat_points = x_dim*y_dim
    for k in operator_set:
        for i in range(lat_points):
            for j in range (lat_points):
                if (i == j+1):
                    if b_rand == True:
                        hamiltonian_list.append(1 * np.matmul(initialize_operator(k, i, lat_points), initialize_operator(k, j, lat_points)))
                        indices.append([i, j])
                    else:
                        alpha = np.random.exponential(scale=0.1)
                        hamiltonian_list.append(alpha * np.matmul(initialize_operator(k, i, lat_points), initialize_operator(k, j, lat_points)))
                        indices.append([i, j])

            if np.array_equal(Z, k) == True:
                if b_rand == True:
                    beta = np.random.exponential() #if we want to randomize the field strength reponse at each site (might be unphysical)
                    hamiltonian_list.append(beta * initialize_operator(k, i, lat_points))
                    indices.append([i])
                else: 
                    hamiltonian_list.append(b_field * initialize_operator(k, i, lat_points))
                    indices.append([i])

    return np.array(hamiltonian_list, indices, length)


def exp_distr_heisenberg_hamiltonian(length, b_field, rng_seed, b_rand):
    #b_rand is a boolean that either sets the field to be randomized or the interactions (if false)
    y_dim = 1
    x_dim = length #restrict to 1d spin change so we can get more disjoint regions
    np.random.seed(rng_seed)
    hamiltonian_list = []
    indices = []
    #graph = initialize_graph(x_dim, y_dim)
    operator_set = [X, Y, Z]
    lat_points = x_dim*y_dim
    for k in operator_set:
        for i in range(lat_points):
            for j in range (lat_points):
                if (i == j+1):
                    if b_rand == True:
                        hamiltonian_list.append(1 * np.matmul(initialize_operator(k, i, lat_points), initialize_operator(k, j, lat_points)))
                        #indices.append([i, j])
                    else:
                        alpha = np.random.exponential(scale=0.1)
                        hamiltonian_list.append(alpha * np.matmul(initialize_operator(k, i, lat_points), initialize_operator(k, j, lat_points)))
                        #indices.append([i, j])

            if np.array_equal(Z, k) == True:
                if b_rand == True:
                    beta = np.random.exponential() #if we want to randomize the field strength reponse at each site (might be unphysical)
                    hamiltonian_list.append(beta * initialize_operator(k, i, lat_points))
                    #indices.append([i])
                else: 
                    hamiltonian_list.append(b_field * initialize_operator(k, i, lat_points))
                    #indices.append([i])

    return np.array(hamiltonian_list)


def ising_model(dim, b_field = 0, rng_seed=1):
    '''Function does not currently have randomization effects, b_field picks the J/g ratio, where J==1'''
    ham_list = []
    for i in range(dim):
        for j in range(dim):
            if ((i==j+1) or (i==0 and j == dim-1)):
                ham_list.append(-1 * np.matmul(initialize_operator(Z, i, dim), initialize_operator(Z, j, dim)))
            
        if b_field != 0:
            ham_list.append(b_field * initialize_operator(X, i, dim))

    return np.array(ham_list)