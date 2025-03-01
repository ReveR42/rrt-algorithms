# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import enum

import random
import numpy as np

from rrt_algorithms.utilities.geometry import steer
from rrt_algorithms.rrt.heuristics import segment_cost, path_cost
from rrt_algorithms.rrt.rrt_star import RRTStar


class Status(enum.Enum):
    FAILED = 1
    TRAPPED = 2
    ADVANCED = 3
    REACHED = 4


class RRTStarBidirectional(RRTStar):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01, rewire_count=None):
        """
        Bidirectional RRT* Search
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        :param rewire_count: number of nearby vertices to rewire
        """
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc, rewire_count)
        self.sigma_best = None  # best solution thus far
        self.c_best = float('inf')  # length of best solution thus far
        self.swapped = False

    def rewire(self, tree, x_new, L_near):
        """
        Rewire tree to shorten edges if possible
        Only rewires vertices according to rewire count
        :param tree: int, tree to rewire
        :param x_new: tuple, newly added vertex
        :param L_near: list of nearby vertices used to rewire
        :return:
        """
        for _, x_near in L_near:
            x_init = self.x_init if tree == 0 else self.x_goal
            curr_cost = path_cost(self.trees[tree].E, x_init, x_near)
            tent_cost = path_cost(self.trees[tree].E, x_init, x_new) + segment_cost(x_new, x_near)
            if tent_cost < curr_cost and self.X.collision_free(x_near, x_new, self.r):
                self.trees[tree].E[x_near] = x_new

    def swap_trees(self):
        """
        Swap trees and start/goal
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

    def extend_star(self, tree, x_rand):
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = steer(x_nearest, x_rand, self.q)
        if self.X.collision_free(x_nearest, x_new, self.r):
            # get nearby vertices and cost-to-come
            x_init = self.x_init if tree == 0 else self.x_goal
            L_near = self.get_nearby_vertices(tree, x_init, x_new)
            # check nearby vertices for total cost and connect shortest valid edge
            x_min = self.connect_shortest_valid(tree, x_new, L_near)

            # rewire tree
            if x_new in self.trees[tree].E:
                try:
                    L_near.remove(x_min)
                except:
                    pass
                self.rewire(tree, x_new, L_near)

            if np.abs(np.sum(np.array(x_new) - np.array(x_rand))) < 1e-2:
                return x_new, Status.REACHED
            return x_new, Status.ADVANCED
        return x_new, Status.TRAPPED

    def connect_star(self, tree, x):
        S = Status.ADVANCED
        while S == Status.ADVANCED:
            x_new, S = self.extend_star(tree, x)
        return x_new, S

    def rrt_star_bidirectional(self):
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)
        self.add_tree()
        self.add_vertex(1, self.x_goal)
        self.add_edge(1, self.x_goal, None)

        while self.samples_taken < self.max_samples:
            x_rand = self.X.sample_free()
            x_new, status = self.extend_star(0, x_rand)
            if status != Status.TRAPPED:
                x_new, connect_status = self.connect_star(1, x_new)
                if connect_status == Status.REACHED:
                    self.unswap()
                    first_part = self.reconstruct_path(0, self.x_init, self.get_nearest(0, x_new))
                    second_part = self.reconstruct_path(1, self.x_goal, self.get_nearest(1, x_new))
                    second_part.reverse()
                    return first_part + second_part
            self.swap_trees()
            self.samples_taken += 1
