# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

# TODO: - save as images
#       - redo rrt* implementation
#       - check why rewiring ineffective
#       - optimize algorithms

import sys

import numpy as np

from rrt_algorithms.rrt.rrt_connect import RRTConnect
from rrt_algorithms.rrt.rrt_star_connect import RRTStarBidirectional
from rrt_algorithms.rrt.informed_rrt_star_connect import InformedRRTStarBidirectional

from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.obstacle_generation import generate_random_obstacles
from rrt_algorithms.utilities.plotting import Plot, draw_map

import matplotlib.pyplot as plt
import matplotlib as mpl

from results.view_results import plot_median_graphs

mpl.use('Qt5Agg')

# number of searches to run for statistical analysis
N, type = None, None

for i, arg in enumerate(sys.argv):
    if arg == '-N':
        N = int(sys.argv[i + 1])
    if arg == '--type':
        type = sys.argv[i + 1]

if N is None:
    N = 1
if type not in ["cluttered", "single"]:
    type = "free"

print(f"Running {N} searches of type {type}")

q = 4  # length of tree edges
r = 0.5  # length of smallest edge to check for intersection with obstacles
max_samples = 3000  # max number of samples to take before timing out
# rewire_count = 32  # optional, number of nearby branches to rewire
rewire_count = None  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal
n = 40  # number of obstacles for random obstacle generation

# Genrate search space
X_dimensions = np.array([(0, 100), (0, 100)])  # dimensions of Search Space

if type == "cluttered":
    x_init = (0, 0)  # starting location
    x_goal = (100, 100)  # goal location
else:
    x_init = (15, 50)  # starting location
    x_goal = (85, 50)  # goal location

# generate obstacles
if type == "cluttered":
    inp = "y"
    while inp == "y":
        X = SearchSpace(X_dimensions)  # create search space
        Obstacles = generate_random_obstacles(X, x_init, x_goal, n)
        draw_map(X, X_dimensions, Obstacles, x_init, x_goal, name="cluttered_map", save_as_img=True)

        inp = input("regenerate obstacles? (y/n): ")
elif type == "single":
    Obstacles = np.array([(30, 30, 70, 70)])  # uncomment to use a single obstacle
    X = SearchSpace(X_dimensions, Obstacles)  # create search space
    draw_map(X, X_dimensions, Obstacles, x_init, x_goal, name="single_map", save_as_img=True)

else:
    X = SearchSpace(X_dimensions)  # create search space
    draw_map(X, X_dimensions, [], x_init, x_goal, name="free_map", save_as_img=True)

# List for statistics
all_iteration_c_bests = []
all_iterations_cpu_times = []

successes = []
runtimes = []
iterations_of_first_solutions = []
path_costs = []
tree_densities = []

print("running searches...")
for k in range(N):
    print(f"Run {k + 1}/{N}")

    # run rrt_searches
    print("RRT Connect")
    rrt_connect = RRTConnect(X, q, x_init, x_goal, max_samples, r, prc)
    conn_path = rrt_connect.rrt_connect()

    print("RRT Star Connect")
    rrt_star_connect = RRTStarBidirectional(
        X, q, x_init, x_goal, max_samples, r, prc, rewire_count)
    star_conn_path = rrt_star_connect.rrt_star_bidirectional()

    print("Informed RRT Star Connect")
    informed_rrt_star_connect = InformedRRTStarBidirectional(
        X, q, x_init, x_goal, max_samples, r, prc, rewire_count)
    inf_star_conn_path = informed_rrt_star_connect.informed_rrt_star_bidirectional()

    # statistics
    successes.append([conn_path is not None, star_conn_path is not None, inf_star_conn_path is not None])
    runtimes.append([rrt_connect.t_max, rrt_star_connect.t_max, informed_rrt_star_connect.t_max])

    rrt_connect_first = rrt_connect.iteration_c_best[0][0] if conn_path is not None else max_samples
    rrt_star_connect_first = rrt_star_connect.iteration_c_best[0][0] if star_conn_path is not None else max_samples
    informed_rrt_star_connect_first = informed_rrt_star_connect.iteration_c_best[0][
        0] if inf_star_conn_path is not None else max_samples
    iterations_of_first_solutions.append([rrt_connect_first, rrt_star_connect_first, informed_rrt_star_connect_first])

    rrt_connect_cost = rrt_connect.c_best if conn_path is not None else np.inf
    rrt_star_connect_cost = rrt_star_connect.c_best if star_conn_path is not None else np.inf
    informed_rrt_star_connect_cost = informed_rrt_star_connect.c_best if inf_star_conn_path is not None else np.inf
    path_costs.append([rrt_connect_cost, rrt_star_connect_cost, informed_rrt_star_connect_cost])

    rrt_connect_density = (rrt_connect.trees[0].V_count, rrt_connect.trees[1].V_count,
                           rrt_connect.trees[0].V_count + rrt_connect.trees[1].V_count)
    rrt_star_connect_density = (rrt_star_connect.trees[0].V_count, rrt_star_connect.trees[1].V_count,
                                rrt_star_connect.trees[0].V_count + rrt_star_connect.trees[1].V_count)
    informed_rrt_star_connect_density = (
        informed_rrt_star_connect.trees[0].V_count, informed_rrt_star_connect.trees[1].V_count,
        informed_rrt_star_connect.trees[0].V_count + informed_rrt_star_connect.trees[1].V_count)
    tree_densities.append([rrt_connect_density, rrt_star_connect_density, informed_rrt_star_connect_density])

    # graph data
    all_iteration_c_bests.append([rrt_connect.iteration_c_best, rrt_star_connect.iteration_c_best,
                                  informed_rrt_star_connect.iteration_c_best])
    all_iterations_cpu_times.append([rrt_connect.iteration_cpu, rrt_star_connect.iteration_cpu,
                                     informed_rrt_star_connect.iteration_cpu])

# plot searches
plot = Plot(f"rrt_connect_{rrt_connect.X.dimensions}d_{type}")
plot.layout.update(dict(xaxis=dict(range=X_dimensions[0]), yaxis=dict(range=X_dimensions[1])))
plot.plot_tree(X, rrt_connect.trees)
if conn_path is not None:
    plot.plot_path(X, conn_path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True, save_as_img=True)

plot = Plot(f"rrt_star_connect_{rrt_star_connect.X.dimensions}d_{type}")
plot.layout.update(dict(xaxis=dict(range=X_dimensions[0]), yaxis=dict(range=X_dimensions[1])))
plot.plot_tree(X, rrt_star_connect.trees)
if star_conn_path is not None:
    plot.plot_path(X, star_conn_path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True, save_as_img=True)

plot = Plot(f"informed_rrt_star_connect_{informed_rrt_star_connect.X.dimensions}d_{type}")
plot.layout.update(dict(xaxis=dict(range=X_dimensions[0]), yaxis=dict(range=X_dimensions[1])))
plot.plot_tree(X, informed_rrt_star_connect.trees)
if inf_star_conn_path is not None:
    plot.plot_path(X, inf_star_conn_path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True, save_as_img=True)

# Format and store plot data

all_c_bests_padded = []
all_cpu_times_padded = []


def format_iteration_cost(iteration_c_best):
    if len(iteration_c_best) == 0:
        return [], []

    # getting iterations
    iterations = range(iteration_c_best[0][0], max_samples)

    # getting best costs
    c_bests = []

    for i in range(len(iteration_c_best) - 1):
        iteration_curr, cost_curr = iteration_c_best[i]
        iterations_next, _ = iteration_c_best[i + 1]
        c_bests.extend([cost_curr for _ in range(iteration_curr, iterations_next)])

    last_iteration, last_cost = iteration_c_best[-1]
    c_bests.extend([last_cost for _ in range(last_iteration, max_samples)])

    return iterations, c_bests


def format_cpu(iteration_cpu):
    if len(iteration_cpu) == 0:
        return [], []

    # getting best costs
    cpu_time = [0]

    for i in range(len(iteration_cpu) - 1):
        iteration_curr, cost_curr = iteration_cpu[i]
        iterations_next, _ = iteration_cpu[i + 1]
        cpu_time.extend([cost_curr for _ in range(iteration_curr, iterations_next)])

    last_iteration, last_cost = iteration_cpu[-1]
    cpu_time.extend([last_cost for _ in range(last_iteration, max_samples)])

    return cpu_time


# Plot graphs
for k in range(N):
    fig = plt.figure(k, figsize=(13, 6))
    ax1, ax2 = fig.subplots(1, 2)

    rrt_connect_iteration_c_best, rrt_star_connect_iteration_c_best, informed_rrt_star_connect_iteration_c_best = \
        all_iteration_c_bests[k]
    iterations_connect, c_bests_connect = format_iteration_cost(rrt_connect_iteration_c_best)
    iterations_star_connect, c_bests_star_connect = format_iteration_cost(rrt_star_connect_iteration_c_best)
    iterations_inf_star_connect, c_bests_inf_star_connect = format_iteration_cost(
        informed_rrt_star_connect_iteration_c_best)

    ax1.set_title('cost vs iteration')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('cost')

    ax1.plot(iterations_connect, c_bests_connect, label='RRT Connect', color='red')
    ax1.plot(iterations_star_connect, c_bests_star_connect, label='RRT* Connect', color='blue')
    ax1.plot(iterations_inf_star_connect, c_bests_inf_star_connect, label='Informed RRT* Connect', color='green')

    all_c_bests_padded.append(
        [np.hstack([[k], np.inf * np.ones(max_samples - len(c_bests_connect)), c_bests_connect]),
         np.hstack([[k], np.inf * np.ones(max_samples - len(c_bests_star_connect)), c_bests_star_connect]),
         np.hstack([[k], np.inf * np.ones(max_samples - len(c_bests_inf_star_connect)), c_bests_inf_star_connect])]
    )

    rrt_connect_iteration_cpu, rrt_star_connect_iteration_cpu, informed_rrt_star_connect_iteration_cpu = \
        all_iterations_cpu_times[k]

    rrt_connect_cpu = format_cpu(rrt_connect_iteration_cpu)
    rrt_star_connect_cpu = format_cpu(rrt_star_connect_iteration_cpu)
    rrt_inf_star_connect_cpu = format_cpu(informed_rrt_star_connect_iteration_cpu)

    ax2.set_title('CPU time vs iteration')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('CPU time (s)')

    ax2.plot(rrt_connect_cpu, label='RRT Connect', color='red')
    ax2.plot(rrt_star_connect_cpu, label='RRT* Connect', color='blue')
    ax2.plot(rrt_inf_star_connect_cpu, label='Informed RRT* Connect', color='green')

    all_cpu_times_padded.append([np.hstack([[k], rrt_connect_cpu]),
                                 np.hstack([[k], rrt_star_connect_cpu]),
                                 np.hstack([[k], rrt_inf_star_connect_cpu])])

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    plt.savefig(f"results/{type}/run_{k}_info.png")
    plt.close(k)

# Plotting and saving graphs
plot_median_graphs(type, max_samples, all_c_bests_padded, all_cpu_times_padded)
plot_median_graphs(type, max_samples, all_c_bests_padded, all_cpu_times_padded, plot_std=True)

# Save data
np.savez_compressed(f"results/{type}/data.npz",
                    c_bests=np.array(all_c_bests_padded),
                    cpu_times=np.array(all_cpu_times_padded),
                    allow_pickle=False)

# Print statistics
success_rates = np.sum(successes, axis=0) / N
runtimes_medians, runtimes_std = np.median(runtimes, axis=0), np.std(runtimes, axis=0)
iterations_of_first_solutions_medians, iterations_of_first_solutions_std = \
    np.median(iterations_of_first_solutions, axis=0), np.std(iterations_of_first_solutions, axis=0)
path_costs_medians, path_costs_std = np.median(path_costs, axis=0), np.std(path_costs, axis=0)
tree_densities_medians, tree_densities_std = np.median(tree_densities, axis=0), np.std(tree_densities, axis=0)

datastring = (f"Results for {N} run(s) of {max_samples} samples with:\n"
              f"  type: {type}\n"
              f"  search space: X_dimensions={X_dimensions.flatten()}, x_init={x_init}, x_goal={x_goal}\n"
              f"  parameters: q={q}, r={r}, rewire_count={rewire_count}, prc={prc}, n={n}\n\n"
              f"Success Rate:\n"
              f"  RRT Connect: {success_rates[0]}\n"
              f"  RRT* Connect: {success_rates[1]}\n"
              f"  Informed RRT* Connect: {success_rates[2]}\n\n"
              f"Runtime:\n"
              f"  RRT Connect: {runtimes_medians[0]} ± {runtimes_std[0]}\n"
              f"  RRT* Connect: {runtimes_medians[1]} ± {runtimes_std[1]}\n"
              f"  Informed RRT* Connect: {runtimes_medians[2]} ± {runtimes_std[2]}\n\n"
              f"Iterations of First Solution:\n"
              f"  RRT Connect: {iterations_of_first_solutions_medians[0]} ± {iterations_of_first_solutions_std[0]}\n"
              f"  RRT* Connect: {iterations_of_first_solutions_medians[1]} ± {iterations_of_first_solutions_std[1]}\n"
              f"  Informed RRT* Connect: {iterations_of_first_solutions_medians[2]} ± {iterations_of_first_solutions_std[2]}\n\n"
              f"Path Cost:\n"
              f"  RRT Connect: {path_costs_medians[0]} ± {path_costs_std[0]}\n"
              f"  RRT* Connect: {path_costs_medians[1]} ± {path_costs_std[1]}\n"
              f"  Informed RRT* Connect: {path_costs_medians[2]} ± {path_costs_std[2]}\n\n"
              f"Tree Densities:\n"
              f"  RRT Connect: (start_tree={tree_densities_medians[0, 0]} ± {tree_densities_std[0, 0]}, goal_tree={tree_densities_medians[0, 1]} ± {tree_densities_std[0, 1]}, total={tree_densities_medians[0, 2]} ± {tree_densities_std[0, 2]}))\n"
              f"  RRT* Connect: (start_tree={tree_densities_medians[1, 0]} ± {tree_densities_std[1, 0]}, goal_tree={tree_densities_medians[1, 1]} ± {tree_densities_std[1, 1]}, total={tree_densities_medians[1, 2]} ± {tree_densities_std[1, 2]}))\n"
              f"  Informed RRT* Connect: (start_tree={tree_densities_medians[2, 0]} ± {tree_densities_std[2, 0]}, goal_tree={tree_densities_medians[2, 1]} ± {tree_densities_std[2, 1]}, total={tree_densities_medians[2, 2]} ± {tree_densities_std[2, 2]}))\n")

print(datastring)

# Save statistics
with open(f"results/{type}/statistics.txt", "w") as file:
    file.write(datastring)
