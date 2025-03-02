# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np

from rrt_algorithms.rrt.heuristics import segment_cost, path_cost, cost_to_go, cost_to_come
from rrt_algorithms.rrt.rrt_star_connect import RRTStarBidirectional
from rrt_algorithms.rrt.rrt_connect import Status

from time import process_time


class InformedRRTStarBidirectional(RRTStarBidirectional):
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
        self.previous_c_best = np.inf
        self.X_soln = []
        self.C = self.rotate_to_world_frame(np.array(self.x_init), np.array(self.x_goal))

    def rotate_to_world_frame(self, x_init, x_goal):
        a1 = (x_goal - x_init) / np.linalg.norm(x_goal - x_init)
        M = a1.reshape(self.X.dimensions, 1) @ np.eye(self.X.dimensions)[np.newaxis, 0]
        U, S, V_T = np.linalg.svd(M)

        D = np.diag(np.concatenate([np.ones(self.X.dimensions - 2), [np.linalg.det(U), np.linalg.det(V_T.T)]]))
        C = U @ D @ V_T

        return C

    def informed_sample(self):
        x_init = np.array(self.x_init)
        x_goal = np.array(self.x_goal)

        if self.c_best is not np.inf:
            c_min = np.linalg.norm(x_goal - x_init)
            x_center = (x_init + x_goal) / 2.

            r = [self.c_best / 2]
            for i in range(1, self.X.dimensions):
                r.append(np.sqrt(self.c_best ** 2 - c_min ** 2) / 2)

            L = np.diag(r)
            x_ball = 2 * np.random.rand(self.X.dimensions) - 1
            x_rand = (self.C @ L @ x_ball).flatten() + x_center
        else:
            x_rand = self.X.sample_free()

        return x_rand

    def calculate_shortest_path(self):
        """
        Calculate shortest path cost
        :return: float, shortest path cost
        """
        c_min = np.inf
        for x in self.X_soln:
            c_tent = path_cost(self.trees[0].E, self.x_init, x) + path_cost(self.trees[1].E, self.x_goal, x)
            if c_tent < c_min:
                c_min = c_tent
                self.x_best = x
        return c_min

    def prune_tree(self, tree, c_best):
        x_init = self.x_init if tree == 0 else self.x_goal
        x_goal = self.x_goal if tree == 0 else self.x_init

        V_prune = [None]
        while len(V_prune) > 0:
            # TODO: iterate through Index properly
            V_prune = [v for v in self.trees[tree].V if cost_to_come(x_init, v) + cost_to_go(x_goal, v) > c_best
                       and v not in self.trees[tree].E.values()]
            for v in V_prune:
                while v in self.trees[tree].E:
                    self.trees[tree].E.pop(v)
                while self.trees[tree].V.count(v + v) > 0:
                    self.trees[tree].V.delete(0, v + v)

    def informed_rrt_star_bidirectional(self):
        t0_process = process_time()

        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)
        self.add_tree()
        self.add_vertex(1, self.x_goal)
        self.add_edge(1, self.x_goal, None)

        percentage_disp = 0
        print(f"{percentage_disp}%")
        while self.samples_taken < self.max_samples:
            if len(self.X_soln) > 0:
                self.previous_c_best = self.c_best
                try:
                    self.c_best = self.calculate_shortest_path()
                except KeyError:
                    self.X_soln.pop(-1)

                if self.c_best < self.previous_c_best:
                    self.iteration_c_best.append((self.samples_taken, self.c_best))

                #     self.prune_tree(0, self.c_best)
                #     self.prune_tree(1, self.c_best)

            x_rand = self.informed_sample()
            x_new, status = self.extend_star(0, x_rand)
            if status != Status.TRAPPED:
                x_new, connect_status = self.connect_star(1, x_new)
                if connect_status == Status.REACHED:
                    # if not (self.trees[0].V.count(x_new) and self.trees[1].V.count(x_new)):
                    # print(f"bingus1 : {self.trees[0].V.count(x_new)} {self.trees[1].V.count(x_new)}")
                    # else:
                    self.X_soln.append(x_new)
            self.swap_trees()
            self.samples_taken += 1
            self.iteration_cpu.append((self.samples_taken, process_time() - t0_process))

            percentage_current = 100 if self.max_samples == 1 else round(
                self.samples_taken / (self.max_samples - 1) * 100.)
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
