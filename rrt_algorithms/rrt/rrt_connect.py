import enum

import numpy as np

from rrt_algorithms.rrt.rrt_base import RRTBase
from rrt_algorithms.utilities.geometry import steer
from rrt_algorithms.rrt.heuristics import path_cost

from time import process_time


class Status(enum.Enum):
    FAILED = 1
    TRAPPED = 2
    ADVANCED = 3
    REACHED = 4


class RRTConnect(RRTBase):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRTConnect planner
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc)
        self.swapped = False

        self.c_best = np.inf
        self.x_best = None
        self.iteration_c_best = []
        self.t_max = 0
        self.iteration_cpu = []

    def swap_trees(self):
        """
        Swap trees only
        """
        # swap trees
        self.trees[0], self.trees[1] = self.trees[1], self.trees[0]
        # swap start/goal
        self.x_init, self.x_goal = self.x_goal, self.x_init
        self.swapped = not self.swapped

    def unswap(self):
        """
        Check if trees have been swapped and unswap
        """
        if self.swapped:
            self.swap_trees()

    def extend(self, tree, x_rand):
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = steer(x_nearest, x_rand, self.q)
        if self.connect_to_point(tree, x_nearest, x_new):
            if np.abs(np.sum(np.array(x_new) - np.array(x_rand))) < 1e-2:
                return x_new, Status.REACHED
            return x_new, Status.ADVANCED
        return x_new, Status.TRAPPED

    def connect(self, tree, x):
        S = Status.ADVANCED
        while S == Status.ADVANCED:
            x_new, S = self.extend(tree, x)
        return x_new, S

    def rrt_connect(self):
        """
        RRTConnect
        :return: set of Vertices; Edges in form: vertex: [neighbor_1, neighbor_2, ...]
        """
        t0_process = process_time()

        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)
        self.add_tree()
        self.add_vertex(1, self.x_goal)
        self.add_edge(1, self.x_goal, None)

        percentage_disp = 0
        print(f"{percentage_disp}%")
        while self.samples_taken < self.max_samples:
            x_rand = self.X.sample_free()
            x_new, status = self.extend(0, x_rand)
            if status != Status.TRAPPED:
                x_new, connect_status = self.connect(1, x_new)
                if connect_status == Status.REACHED:
                    try:
                        c_tent = path_cost(self.trees[0].E, self.x_init, x_new) + path_cost(self.trees[1].E,
                                                                                            self.x_goal, x_new)
                        if c_tent < self.c_best:
                            self.x_best = x_new
                            self.c_best = c_tent
                            self.iteration_c_best.append((self.samples_taken, self.c_best))
                    except KeyError:
                        pass

            self.swap_trees()
            self.samples_taken += 1
            self.iteration_cpu.append((self.samples_taken, process_time() - t0_process))

            percentage_current = 100 if self.max_samples == 1 else int(self.samples_taken / (self.max_samples - 1) * 100.)
            if percentage_disp != percentage_current:
                percentage_disp = percentage_current
                print(f"{percentage_disp}%")

        self.t_max = process_time() - t0_process

        if self.x_best is None:
            return None

        self.unswap()
        first_part = self.reconstruct_path(0, self.x_init, self.get_nearest(0, self.x_best))
        second_part = self.reconstruct_path(1, self.x_goal, self.get_nearest(1, self.x_best))
        second_part.reverse()

        print(f"Best path cost history: {self.iteration_c_best}")

        return first_part + second_part
