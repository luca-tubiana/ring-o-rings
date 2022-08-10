"""
Implement a class to build distorted dodecagons for an hexagonal lattice of linked rings.
"""
from itertools import product

import numpy as np


class Dodecagon:
    def __init__(self, radius, nbeads):
        self.n = 12
        self.r = radius
        theta = np.pi / 6
        self.edge = self.r * np.sqrt(2 - np.sqrt(3))
        self.nbeads = nbeads
        self.vertices = radius * np.array([[np.sin(i * theta), np.cos(i * theta), 0] for i in range(self.n)])
        self.distorted_vertices = self._create_all_distorted()
        self.regular_12gon = self._add_beads(self.vertices, self.nbeads)
        self.distorted_12gons = np.array([self._add_beads(x, self.nbeads) for x in self.distorted_vertices])

    def _add_beads(self, coords, nbeads):
        assert nbeads >= 36, "excluded volume requires nbeads >= 36"
        assert nbeads % 12 == 0, "we need multiples of 12"
        nseg = nbeads // 12
        cb = np.repeat(coords, nseg, axis=0)
        dl = np.diff(coords, axis=0, append=[coords[0]]) / nseg
        for i in range(self.n):
            for j in range(nseg):
                cb[i * nseg + j] += j * dl[i]
        return cb

    def _create_all_distorted(self):
        n = self.n
        # dz comes from the formulae of a regular dodecagon
        dz = self.edge * np.sin(np.pi / 12)
        idx = np.array([1, 3, 5, 7, 9, 11])
        all_vertices = []
        for pattern in product([-1, 1], repeat=3):
            c = np.copy(self.vertices)
            for i in [0, 1, 2]:
                j = idx[2 * i]
                k = idx[2 * i + 1]
                c[j][2] = pattern[i] * dz
                c[j][0] = 0.5 * (c[j - 1][0] + c[j + 1][0])
                c[j][1] = 0.5 * (c[j - 1][1] + c[j + 1][1])
                c[k][2] = - pattern[i] * dz
                c[k][0] = 0.5 * (c[k - 1][0] + c[(k + 1) % n][0])
                c[k][1] = 0.5 * (c[k - 1][1] + c[(k + 1) % n][1])
            all_vertices.append(c)
        return np.array(all_vertices)
