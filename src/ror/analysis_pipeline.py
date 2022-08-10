# import numpy as np
import re
from pathlib import Path

import pandas as pd

from ror.IO import reconstruct_traj
from ror.backbone_analyses import CatLocProp, CatGlobProp
from ror.com_CG import CatComCG


class AnalysisPipeline:
    """
    Performs the whole analysis of a trajectory starting from lammps data.
    The pipeline proceeds as follows
    1. Coarse-graining of the system
    2. Computing local quantities (per elementary ring, e.g. local twist)
    3. Compute global quantities (per catenane, e.g. Rg2, writhe)
    """

    def __init__(self, input_dir, cg=CatComCG, verse=1, twist_normals=False):
        """
        Initialize the analysis and in particular the coarse-graining procedure
        Parameters
        ----------
        input_dir: input directory storing the lammps trajectory with .lammpstrj  extension
        cg: Type of coarse graining
        verse: 1 - use the orientation provided by bead ordering. -1 - reverse it.
        twist_normals: if True reassign the normals to match those of a twisted ribbon:
        each alternate ring flips the normals of all subsequent rings.
        """
        self.simdir = Path(input_dir)
        m_rings, n_beads, lk = AnalysisPipeline.get_details(input_dir)
        self.m = m_rings
        self.n = n_beads
        self.lk = lk
        self.m_a = int(lk * m_rings) // 2
        self._traj = None
        self._cat = None
        self._loc = None
        self._global = None
        self.cg = cg  # coarse graining method
        self.twn = twist_normals
        self.verse = verse

    def __str__(self):
        return f"M{self.m}n{self.n}Malt{self.m_a}"

    def __repr__(self):
        return f"AnalysisPipeline({self.simdir})"

    @staticmethod
    def get_details(input_dir):
        input_dir = Path(input_dir)
        els = re.split('[MmnNLlkK]', input_dir.name)
        m = int(els[1])
        n = int(els[2])
        lk = float(els[-1])
        return m, n, lk

    def __mkoutdir(self, outdir, flag=True):
        outdir = Path(outdir)
        sysname = self.__str__()
        (outdir / sysname).mkdir(parents=True, exist_ok=flag)
        return outdir / sysname

    @property
    def traj(self):
        if self._traj is None:
            self._traj = reconstruct_traj(self.simdir.glob('traj*lammpstrj'))
        return self._traj

    @property
    def cat(self):
        if self._cat is None:
            self._cat = self.cg(self.traj, m_a=self.m_a, verse=self.verse, twist_normals=self.twn)
        return self._cat

    @property
    def local_properties(self):
        if self._loc is None:
            self._loc = CatLocProp(self.cat.cgdata)
            self._loc = self._loc.cat_data
        return self._loc

    @property
    def global_properties(self):
        if self._global is None:
            self._global = CatGlobProp(self.local_properties)
            self._global = self._global.data
        return self._global

    def load_local_properties(self, input_dir):
        fname = self.__str__() + '_local_properties.csv'
        self._loc = pd.read_csv(input_dir / fname)

    def save_vtf_file(self, outdir):
        fname = self.__str__() + '_cg.vtf'
        outdir = self.__mkoutdir(outdir)
        cat = self.cat
        cat.to_vtf_file(outdir / fname)

    def save_local_properties(self, outdir):
        fname = self.__str__() + '_local_properties.csv'
        outdir = self.__mkoutdir(outdir)
        loc = self.local_properties
        loc.to_csv(outdir / fname)

    def save_global_properties(self, outdir):
        fname = self.__str__() + '_global_properties.csv'
        outdir = self.__mkoutdir(outdir)
        glb = self.global_properties
        glb.to_csv(outdir / fname)
