import matplotlib
import csv
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot(name_file, name_graph):
    data = np.load(name_file)
    difference = [y - x for x, y in zip(data, data[1:])]
    abs_difference = np.abs(difference)
    axes = plt.gca()
    axes.set_ylim([0.0, 0.5])
    plt.xlabel("Simulation Iteration")
    plt.ylabel("Q-value Difference")
    plt.title(name_graph)
    plt.plot(abs_difference)
    plt.savefig(name_graph)
