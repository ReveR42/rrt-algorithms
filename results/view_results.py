import matplotlib.pyplot as plt
import numpy as np

def plot_median_graphs(type, max_samples, all_c_bests_padded, all_cpu_times_padded, plot_std=False, show=False, save=True, figname="all_info"):
    fig = plt.figure(figname, figsize=(13, 6))
    ax1, ax2 = fig.subplots(1, 2)

    all_iteration_c_bests_median = np.median(all_c_bests_padded, axis=0)[:, 1:]
    all_iteration_c_bests_std = np.std(all_c_bests_padded, axis=0)[:, 1:]

    start_index_conn, start_index_star_conn, start_index_inf_star_conn = \
        np.sum(np.isinf(all_iteration_c_bests_median), axis=1).astype(int)

    ax1.set_title('cost vs iteration')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('cost')

    ax1.plot(list(range(start_index_conn, max_samples)),
             all_iteration_c_bests_median[0, start_index_conn:],
             label='RRT Connect (median)', color='red')
    if plot_std:
        ax1.plot(list(range(start_index_conn, max_samples)),
                 all_iteration_c_bests_median[0, start_index_conn:] + all_iteration_c_bests_std[0, start_index_conn:],
                 label='RRT Connect (std)', linestyle='dashed', color='red', linewidth=0.5)
        ax1.plot(list(range(start_index_conn, max_samples)),
                 all_iteration_c_bests_median[0, start_index_conn:] - all_iteration_c_bests_std[0, start_index_conn:],
                 linestyle='dashed', color='red', linewidth=0.5)

    ax1.plot(list(range(start_index_star_conn, max_samples)),
             all_iteration_c_bests_median[1, start_index_star_conn:],
             label='RRT* Connect (median)', color='blue')
    if plot_std:
        ax1.plot(list(range(start_index_star_conn, max_samples)),
                 all_iteration_c_bests_median[1, start_index_star_conn:] + all_iteration_c_bests_std[1, start_index_star_conn:],
                 label='RRT* Connect (std)', linestyle='dashed', color='blue', linewidth=0.5)
        ax1.plot(list(range(start_index_star_conn, max_samples)),
                 all_iteration_c_bests_median[1, start_index_star_conn:] - all_iteration_c_bests_std[1, start_index_star_conn:],
                 linestyle='dashed', color='blue', linewidth=0.5)

    ax1.plot(list(range(start_index_inf_star_conn, max_samples)),
             all_iteration_c_bests_median[2, start_index_inf_star_conn:],
             label='Informed RRT* Connect (median)', color='green')
    if plot_std:
        ax1.plot(list(range(start_index_inf_star_conn, max_samples)),
                 all_iteration_c_bests_median[2, start_index_inf_star_conn:] + all_iteration_c_bests_std[2,
                                                                               start_index_inf_star_conn:],
                 label='Informed RRT* Connect (std)', linestyle='dashed', color='green', linewidth=0.5)
        ax1.plot(list(range(start_index_inf_star_conn, max_samples)),
                 all_iteration_c_bests_median[2, start_index_inf_star_conn:] - all_iteration_c_bests_std[2,
                                                                               start_index_inf_star_conn:],
                 linestyle='dashed', color='green', linewidth=0.5)

    all_iteration_cpu_times_median = np.median(all_cpu_times_padded, axis=0)[:, 1:]
    all_iteration_cpu_times_std = np.std(all_cpu_times_padded, axis=0)[:, 1:]

    ax2.set_title('CPU time vs iteration')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('CPU time (s)')

    ax2.semilogy(list(range(max_samples)), all_iteration_cpu_times_median[0],
                 label='RRT Connect (median)', color='red')
    if plot_std:
        ax2.semilogy(list(range(max_samples)), all_iteration_cpu_times_median[0] + all_iteration_cpu_times_std[0],
                     label='RRT Connect (std)', linestyle='dashed', color='red', linewidth=0.5)
        ax2.semilogy(list(range(max_samples)), all_iteration_cpu_times_median[0] - all_iteration_cpu_times_std[0],
                     linestyle='dashed', color='red', linewidth=0.5)

    ax2.semilogy(list(range(max_samples)), all_iteration_cpu_times_median[1],
                 label='RRT* Connect (median)', color='blue')
    if plot_std:
        ax2.semilogy(list(range(max_samples)), all_iteration_cpu_times_median[1] + all_iteration_cpu_times_std[1],
                     label='RRT* Connect (std)', linestyle='dashed', color='blue', linewidth=0.5)
        ax2.semilogy(list(range(max_samples)), all_iteration_cpu_times_median[1] - all_iteration_cpu_times_std[1],
                     linestyle='dashed', color='blue', linewidth=0.5)

    ax2.semilogy(list(range(max_samples)), all_iteration_cpu_times_median[2],
                 label='Informed RRT* Connect (median)', color='green')
    if plot_std:
        ax2.semilogy(list(range(max_samples)), all_iteration_cpu_times_median[2] + all_iteration_cpu_times_std[2],
                     label='Informed RRT* Connect (std)', linestyle='dashed', color='green', linewidth=0.5)
        ax2.semilogy(list(range(max_samples)), all_iteration_cpu_times_median[2] - all_iteration_cpu_times_std[2],
                     linestyle='dashed', color='green', linewidth=0.5)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    if show:
        plt.show()
    if save:
        plt.savefig("results/" * (not __name__=="__main__") + f"{type}/all_run_info" + plot_std * "_with_std" + ".png")
    plt.close(figname)

def plot_macro(type, plot_std=False, show=False):

    all_data_cluttered = np.load(f"{type}/data.npz")
    all_c_bests_cluttered_padded, all_cpu_times_cluttered_padded = all_data_cluttered["c_bests"], all_data_cluttered["cpu_times"]

    max_samples = 0
    with open(f"{type}/statistics.txt", "r") as f:
        max_samples = int(f.readline().split(" ")[5])

    plot_median_graphs(type,max_samples, all_c_bests_cluttered_padded, all_cpu_times_cluttered_padded,
                       plot_std=plot_std, figname="all_info", show=show)

if __name__ == "__main__":
    plot_macro("cluttered", plot_std=True)
    plot_macro("cluttered", plot_std=False)
    plot_macro("single", plot_std=True)
    plot_macro("single", plot_std=False)
