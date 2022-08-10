import numpy as np
import pandas as pd


class CatCG:
    """
    Base class for catenane Coarse-Graining. Defines interface.
    """
    conf_id = 'frame'
    ring_id = 'mol_id'
    coord_lab = ['x', 'y', 'z']
    vert_lab = ['cx', 'cy', 'cz']
    tan_lab = ['tx', 'ty', 'tz']
    norm_lab = ['nx', 'ny', 'nz']

    def __init__(self, cat_traj, m_a, verse=1, twist_normals=False):
        assert {CatCG.ring_id, 'x', 'y', 'z'} <= set(cat_traj.columns.to_list())
        if verse > 0:
            self.verse = 1.
        else:
            self.verse = -1
        self.M = cat_traj.groupby(CatCG.conf_id)[CatCG.ring_id].tail()
        self.n = cat_traj.groupby(CatCG.conf_id)[CatCG.ring_id].count() // self.M
        self.m_a = m_a
        self.coords = cat_traj
        self._vertices = None
        self._edges = None
        self._normals = None
        self.twn = twist_normals
        self.n_rings = int(cat_traj.iloc[-1][CatCG.ring_id].max())
        self.n_frames = len(cat_traj.index.get_level_values('frame').unique())
        print(self.n_rings)
        self.prefactors = CatCG._set_normals_prefactors(self.n_rings, self.m_a)
        self.cgdata = pd.concat([self.vertices, self.edges, self.normals], axis=1)

    @property
    def vertices(self):
        if self._vertices is None:
            self._vertices = self._get_vertices()
            self._vertices.rename(columns={l: CatCG.vert_lab[i] for i, l in enumerate(CatCG.coord_lab)}, inplace=True)
        return self._vertices

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self.vertices.groupby([CatCG.conf_id]).apply(self.ring_edges_df)
        return self._edges

    @property
    def normals(self):
        if self._normals is None:
            normals = self.coords.groupby([CatCG.conf_id, CatCG.ring_id])[CatCG.coord_lab].apply(self.ring_normal_df)
            self._normals = normals.rename(columns={l: CatCG.norm_lab[i] for i, l in enumerate(CatCG.coord_lab)})
            self._normals *= self.verse  # corrects the wrong orientation of the rings!!
            if self.twn:
                df = self._normals
                df['norm_pref'] = np.tile(self.prefactors, self.n_frames)
                df[CatCG.norm_lab] = df[CatCG.norm_lab].multiply(df['norm_pref'], axis='index')
        return self._normals

    def _set_normals_prefactors(n_rings, m_a):
        ids = np.arange(n_rings)
        # tmp = ids[1:-1]
        # if np.random.rand() > 0.5:
        v = ids[ids % 2 == 0]
        # else:
        #    v = tmp[tmp % 2 == 1]
        # v = np.sort(np.random.choice(v, Malt, replace=False))
        # v = np.append(v, len(ids))
        prefactors = np.ones(n_rings)
        # for i in range(0, len(v[:-1]), 2):
        #    prefactors[v[i]:v[i + 1]] *= -1
        for x in v[:m_a]:
            prefactors[x:] *= -1
        return prefactors

    def _get_vertices(self):
        return NotImplementedError

    @staticmethod
    def ring_edges_df(ring: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ring edges.

        Parameters
        ----------
        ring: pd.DataFrame
            must contain columns 'x', 'y','z'

        Returns
        -------

        """
        xyz = ring[CatCG.vert_lab].values - ring[CatCG.vert_lab].values.mean()
        xyz_r = np.roll(xyz, -1, axis=0)
        t = xyz_r - xyz
        return pd.DataFrame(t, columns=CatCG.tan_lab, index=ring.index)

    @staticmethod
    def ring_normal_df(ring: pd.DataFrame) -> pd.Series:
        """
        Compute the average normal for a ring
        :param ring:  pd.DataFrame with columns 'x','y','z'
        :return: pd.Series with coordinates of the average normal
        """
        ld = ring.shape[0] // 4
        xyz = ring[CatCG.coord_lab].values
        xyz_r = xyz - xyz.mean(axis=0)
        n = np.cross(xyz_r[:3 * ld], xyz_r[ld:]).mean(axis=0)
        n /= np.linalg.norm(n)
        return pd.Series(n, index=CatCG.coord_lab)

    def to_knt_file(self, filename):
        """
        Write a .knt file to be analyzed with entang
        """
        data = self.vertices
        colx, coly, colz = CatCG.vert_lab
        n_atn_str = "{:10d}\n"
        at_str = "{:16.9f} {:16.9f} {:16.9f}\n"
        with open(filename, 'w') as fout:
            for ts, new in data.groupby(level=CatCG.conf_id):
                data = new.droplevel(CatCG.conf_id).reset_index()
                fout.write(n_atn_str.format(data.shape[0] + 1))
                data.reset_index()
                for index, row in data.iterrows():
                    fout.write(at_str.format(row[colx], row[coly], row[colz]))
                row = data.iloc[0]
                fout.write(at_str.format(row[colx], row[coly], row[colz]))

    def to_vtf_file(self, filename):
        """
        Write a .vtf file to be opened with VMD
        Note: assumes that all frames have the same value of M
        """
        data = self.cgdata
        colx, coly, colz = CatCG.vert_lab
        nx, ny, nz = CatCG.norm_lab
        at_str = "{:d} {:16.9f} {:16.9f} {:16.9f}\n"
        M = self.M.values[-1]
        s = 0.5 * self.n.values[-1] / np.pi
        with open(filename, 'w') as fout:
            # write preamble (topology)
            fout.write(f"atom default name CENTER radius 1.0\n")
            max_line_len = 20
            iters = M // max_line_len
            # normals atoms
            for i in range(iters):
                idx0 = i * max_line_len
                fout.write(f"atom {2 * idx0 + 1}")
                for l in range(1, max_line_len):
                    idx = idx0 + l
                    if idx >= M:
                        break
                    fout.write(f',{2 * idx + 1}')
                fout.write(f" name NORMAL radius 0.5\n")
            # bonds
            for i in range(iters):
                idx0 = i * max_line_len
                fout.write(f"bonds {2 * idx0}:{2 * (idx0 + 1)},{2 * idx0}:{2 * idx0 + 1}")
                for l in range(1, max_line_len):
                    idx = idx0 + l
                    if idx >= M:
                        break
                    fout.write(f',{2 * idx}:{2 * (idx + 1)},{2 * idx}:{2 * idx + 1}')
                fout.write(f"\n")
            # write coordinates
            for ts, new in data.groupby(level=CatCG.conf_id):
                data = new.droplevel(CatCG.conf_id).reset_index()
                # data.reset_index()
                fout.write("timestep indexed\n")
                for index, row in data.iterrows():
                    fout.write(at_str.format(2 * index, row[colx], row[coly], row[colz]))
                    fout.write(at_str.format(2 * index + 1, row[colx] + s * row[nx], row[coly] + s * row[ny],
                                             row[colz] + s * row[nz]))


class CatCGLoader(CatCG):
    """
    Create a CatCG class from a precomputed DataFrame
    """

    def __init__(self, cgdata: pd.DataFrame):
        self.cgdata = cgdata.set_index([CatCG.conf_id, CatCG.ring_id])
        self._vertices = self.cgdata[['cx', 'cy', 'cz']]
        self._normals = self.cgdata[['nx', 'ny', 'nz']]
        self._edges = self.cgdata[['tx', 'ty', 'tz']]
        self._twist = self.cgdata['twist']
        self._stacking = self.cgdata['stacking']
        self._rescaled = self._vertices[['cx', 'cy', 'cz']] - self._vertices.groupby(CatCG.conf_id)[
            ['cx', 'cy', 'cz']].mean()
