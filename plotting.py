import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt

#finds the advantage of using comp over QD and trotter at their respective crossover and plots this
def crossover_advantage(qd_data, trot_data, comp_data, times):
    plt.plot(times,qd_data)
    plt.plot(times,trot_data)
    plt.plot(times, comp_data)

    line_1 = LineString(np.column_stack((times, qd_data)))
    line_2 = LineString(np.column_stack((times, trot_data)))
    intersection1 = line_1.intersection(line_2)
    x1,y1 = intersection1.xy
    plt.plot(*intersection1.xy, 'ro')

    x_intercept = []
    for i in range(len(times)):
        x_intercept.append(x1)
    y_intercept = np.linspace(0, 50000, len(times))
    
    line_3 = LineString(np.column_stack(times, comp_data))
    line_4 = LineString(np.column_stack(x_intercept, y_intercept))

    intersection2 = line_3.intersection(line_4)
    x2, y2 = intersection2.xy
    plt.plot(*intersection2.xy, 'ro')

    comp_adv = y1/y2
    return comp_adv

#plots the distribution of norms in the hamiltonian
def ham_spec(hamiltonian_list):
    norms = []
    index = []
    zero_norms = 0
    for i in range(len(hamiltonian_list)):
        h = hamiltonian_list[i]
        spec = np.linalg.norm(h, ord=2)
        norms.append(spec)
        index.append(i)
        if spec == 0:
            zero_norms += 1
    norms.sort()
    plt.figure(figsize=(9,7))
    plt.plot(index, norms, 'o-')
    plt.xlabel("Index", size = 18)
    plt.ylabel("Spectral Norm", size = 18)
    plt.show()
    print("There are " + str(zero_norms) + " terms with 0 spectral norm")
    return norms