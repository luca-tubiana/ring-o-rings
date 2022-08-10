"""
Implements classes and functions to build poly[n]-catenanes storing twist, create lammps simulation files.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from scipy.spatial.transform import Rotation as R

from ror.dodecagons import Dodecagon

builderdir = Path(os.path.dirname(os.path.abspath(__file__)))


class Catenane:
    def __init__(self, system: pd.DataFrame):
        """Simple class implementing an annular catenane.

        Parameters
        ----------
        system: pd.DataFrame
            data about the system. Must contain columns: 'x','y','z', 'mol_id', 'at_id', 'at_type'...
        """
        self.data = system
        self.n = int(system.groupby('mol_idx')['at_idx'].count().mean())
        self.M = system.mol_idx.nunique()
        self._rc = None
        self._rr = None

    @property
    def rc(self):
        """
        Returns
        -------
        float
            radius of the catenane
        """
        if self._rc is None:
            coords = self.data[['x', 'y', 'z']].values
            opp_coords = np.roll(coords, -coords.shape[0] // 2, axis=0)
            diam = np.linalg.norm(coords - opp_coords, axis=1).max()
            self._rc = diam / 2
        else:
            return self._rc

    @property
    def rr(self):
        """
        Radius of the FIRST ring in the catenane, not the average radius.
        For our purpose, this is enough... still DO NOT USE THIS FOR ANALYSIS!!

        Returns
        -------
        float
            radius of the first ring in the catenane
        """
        if self._rr is None:
            coords = self.data[self.data.mol_id == 1][['x', 'y', 'z']].values
            opp_coords = np.roll(coords, -coords.shape[0] // 2, axis=0)
            diam = np.linalg.norm(coords - opp_coords, axis=1).max()
            self._rr = diam / 2
        else:
            return self._rr


class CatenaneBuilder:
    def __init__(self, n_beads: int, m_distorted: int):
        """Class to build circular Catenanes with given size and different linking numbers.

        Parameters
        ----------
        n_beads : int
            number of beads per ring
        m_distorted : int
            number of distorted rings to control Lk. The catenane will have twice those rings.
        """
        self.M = 2 * m_distorted
        self.M_dis = m_distorted
        self.n = n_beads
        self.Rc = 1.1 * m_distorted * n_beads / (4 * np.pi)
        self.rr = 0.5 * n_beads / np.pi

    def build_system(self, frac_alt: float) -> Catenane:
        """Build an  annular catenane with Lk = frac_alt
        giving n_tw = frac_alt * M_distorted

        Parameters
        ----------
        frac_alt: float
            0<= frac_alt <=1 fraction of alternating distorted rings.

        Returns
        -------
        Catenane
            a Catenane class with the proper fraction of alternating and non-alternating distorted rings.
        """
        system = build_catenane(self.Rc, self.rr, self.n, self.M_dis, frac_alt)
        add_topology(system, inplace=True)
        add_types(system, inplace=True)
        return Catenane(system)


class LammpsLangevinInput:
    template_dir = builderdir / 'templates'
    template_lmp_input = 'template_rings.lmp'

    def __init__(self, bending, steps, n_thermo, n_dump, n_balance, temp=1.0, tau_damp=1.0, seed=None):
        """Lammps data for a Langevin simulation

        Parameters
        ----------
        bending : float
           bending  constant for cosine angle interaction
        steps : int
            simulation steps
        n_thermo : int
            print thermo every n_thermo steps
        n_balance : int
            balance processors every n_balance steps
        n_dump : int
            dump every n_dump steps
        temp : float
            Temperature of the simulation (LJ units)
        tau_damp : float
            damping for Langevine Thermostat
        seed : int
            PRNG seed
        """
        self.bending = bending
        self.steps = int(steps)
        self.n_balance = int(n_balance)
        self.n_thermo = int(n_thermo)
        self.n_restart = min(self.steps // 1000, 1e7)
        self.n_cdump = int(n_dump)
        self.temp = temp
        self.tau_damp = tau_damp
        if seed is None:
            seed = int.from_bytes(os.urandom(3), 'little')
        self.seed = seed

    def to_file(self, datafile, outfile):
        """Generate Lammps input file 'outfile'

        Parameters
        ----------
        datafile : str
            Filename (path) of lammps data file
        outfile : str
            Filename (path) of the file to write.
        """
        values = vars(self)
        values['input'] = datafile
        values['baseinput'] = os.path.basename(values['input'])
        file_loader = FileSystemLoader(LammpsLangevinInput.template_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(LammpsLangevinInput.template_lmp_input)
        with open(outfile, 'w') as f:
            f.write(template.render(**vars(self)))


class LammpsDatafile:
    template_dir = builderdir / 'templates'
    template_lmp_data = 'template_lmp_input.dat'

    def __init__(self, catenane: Catenane, scriptname="ror.LammpsSimBuilder", delta_box=10):
        """Write Lammps input files to simulate a catenane system.

        Parameters
        ----------
        catenane : Catenane
            The annular catenane system to simulate
        scriptname : str
            The name of the script generating the lammps files
        """
        self.catenane = catenane
        self.scriptname = scriptname
        self._box = LammpsDatafile.box_builder(self.catenane, delta=delta_box)
        self._lammps_data = None

    @property
    def box(self):
        return self._box

    @property
    def lammps_data(self):
        if self._lammps_data is None:
            self._lammps_data = self._set_lammps_data()
        return self._lammps_data

    @staticmethod
    def box_builder(catenane: Catenane, delta=10):
        """Build a cubic box around a catenane.

        Parameters
        ----------
        catenane : Catenane
            system to be contained in the box
        delta : float
            extra space for each side (in units of sigma)

        Returns
        -------
        np.array
            box sizes [[-dx,dx], [-dy,dy], [-dz,dz]]
        """
        n = catenane.n
        M = catenane.M
        d_ring = n / np.pi
        d_cat = M * d_ring / 2
        dx = d_cat + 2 * delta
        return 0.5 * np.array([[-dx, dx], [-dx, dx], [-dx, dx]])

    def _set_lammps_data(self) -> dict:
        """Generates dictionary to fill datafile jinja2 template

        Returns
        -------
        Dict
            Dictionary passed to jinja2 to write the data file
        """
        system = self.catenane.data
        atom_string = "{:7d} {:6d} {:6d}   {:16.9f} {:16.9f} {:16.9f}"
        atoms_lines = system.apply(
            lambda row: atom_string.format(*row[['at_idx', 'mol_idx', 'at_type', 'x', 'y', 'z']]), axis=1)
        bonds_lines = system.at_idx.astype(str) + ' ' + system.bond_type.astype(str) + ' ' + system.bonds
        angles_lines = system.at_idx.astype(str) + ' ' + system.angle_type.astype(str) + ' ' + system.angles
        lammps_data = {'script': self.scriptname, 'date': datetime.today().strftime('%Y-%m-%d'),
                       'n_atoms': system.shape[0],
                       'n_bonds': system.shape[0],
                       'n_angles': system.shape[0],
                       'n_atype': 1, 'n_btype': 1, 'n_antype': 1,
                       'xlo': self.box[0][0], 'xhi': self.box[0][1],
                       'ylo': self.box[1][0], 'yhi': self.box[1][1],
                       'zlo': self.box[2][0], 'zhi': self.box[2][1],
                       'atoms': atoms_lines,
                       'bonds': bonds_lines,
                       'angles': angles_lines}
        return lammps_data

    def to_file(self, outfile):
        """Generate Lammps datafile 'outfile'

        Parameters
        ----------
        outfile : str
            Filename (path) of the file to write.
        """
        file_loader = FileSystemLoader(LammpsDatafile.template_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(LammpsDatafile.template_lmp_data)
        with open(outfile, 'w') as f:
            f.write(template.render(**self.lammps_data))


def build_circle(radius: float, nbeads: int) -> np.array:
    """Build a circle of given radius with `nbeads` points."""
    theta = 2 * np.pi / nbeads
    return radius * np.array([[np.sin(i * theta), np.cos(i * theta), 0] for i in range(nbeads)])


def place_on_ngon(radius: float, n_sides: int, objs: Sequence, ind=None) -> pd.DataFrame:
    """
    Place n_sides objects on the vertex on a n-gon with n_sides.
    Objects must have the same length.
    """
    theta = 2 * np.pi / n_sides
    vertices = radius * np.array([[(np.cos(i * theta), np.sin(i * theta), 0)] for i in range(n_sides)])
    objs = np.array(objs)
    assert len(objs.shape) == 3, "objs must contain a list of coordinates (must have 3 axes)"
    l = objs.shape[0]
    if ind is not None:
        assert len(ind) < n_sides + 0.001, f"too many placement indices {len(ind)} > {n_sides}"
    else:
        ind = np.arange(n_sides)
    ind_list = np.repeat(ind, objs.shape[1])
    coords = objs + vertices[ind]
    coords = coords.reshape((coords.shape[0] * coords.shape[1], coords.shape[2]))
    df = pd.DataFrame(np.c_[coords, ind_list], columns=['x', 'y', 'z', 'place'])
    df.place = df.place.astype(int)
    return df


def build_catenane(rc: float, rr: float, nbeads: int, n_distorted: int, frac_alt: float) -> pd.DataFrame:
    """
    Build a catenane with 2*n_distorted sides, alternating circles with distorted rings.
    The total linking number can be controlled through frac_alt, which fixes the number
    of rings contributing +2 to Lk
    """
    # system properties
    alpha = 2 * np.pi / n_distorted
    n_alt = int(n_distorted * frac_alt)
    # rotation matrices
    rot_alpha = R.from_euler('z', [alpha * i for i in range(n_distorted)])  # rotation matrices for rings
    rot_all = R.from_euler('z', alpha / 2)  # rotation for the whole structure
    # checks
    assert frac_alt <= 1, "Fraction can not be larger than 1"
    # assert rr >  Rc * np.sin(alpha/2), "constituents must be linked to first neighbours"
    assert rr < rc * np.sin(alpha), "constituents must not link second neighbours"
    # build constituents
    circle = build_circle(rr, nbeads)
    dod = Dodecagon(rr, nbeads)
    dod_same = dod.distorted_12gons[0]
    dod_alt = dod.distorted_12gons[2]
    # rotate distorted rings to match the edges of the n-gon
    rotated_same = np.array([r.apply(dod_same) for r in rot_alpha])
    rotated_alt = np.array([r.apply(dod_alt) for r in rot_alpha])
    # assign different positions on the n-gon
    ind_alt = np.arange(n_alt)
    ind_same = np.arange(n_alt, n_distorted)
    # build sub-components
    placed_alt = place_on_ngon(rc, n_distorted, rotated_alt[ind_alt], ind_alt)
    placed_same = place_on_ngon(rc, n_distorted, rotated_same[ind_same], ind_same)
    placed_circles = place_on_ngon(rc, n_distorted, np.tile(circle, (n_distorted, 1, 1)))
    # rotate the components with circles
    placed_circles[['x', 'y', 'z']] = rot_all.apply(placed_circles[['x', 'y', 'z']])
    # store characteristic for easy visualization
    placed_alt['type'] = 'alternate'
    placed_same['type'] = 'same'
    placed_circles['type'] = 'circle'
    # shift place indeces
    placed_circles.place = placed_circles.place * 2 + 1
    placed_alt.place *= 2
    placed_same.place *= 2
    # concatenate sub-components and sort according to their indices
    catenane = pd.concat([placed_alt, placed_same, placed_circles])
    # catenane.sort_values(by='place', ascending=True, inplace=True)
    catenane = catenane.rename_axis('Idx').sort_values(by=['place', 'Idx'], ascending=[True, True])
    catenane = catenane.reset_index().drop(columns='Idx')
    catenane['at_idx'] = catenane.index + 1
    catenane.rename(columns={'place': 'mol_idx'}, inplace=True)
    catenane.mol_idx += 1
    return catenane


def add_topology(cat_in: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Add bonds and angles to a catenane DataFrame
    """
    if inplace:
        catenane = cat_in
    else:
        catenane = cat_in.copy()
    nbeads = catenane.groupby('mol_idx').size()
    # auxiliary vectors to create bond and angles within rings of size nbeads.
    v = np.ones(catenane.shape[0], int)
    v[nbeads.cumsum() - 1] = -nbeads + 1
    v2 = v.copy() + 1
    v2[nbeads.cumsum() - 2] = -nbeads + 2
    catenane['bonds'] = catenane.at_idx.astype(str) + ' ' + (catenane.at_idx + v).astype(str)
    catenane['angles'] = catenane.bonds + ' ' + (catenane.at_idx + v2).astype(str)
    return catenane


def add_types(cat_in: pd.DataFrame, inplace=False) -> pd.DataFrame:
    if inplace:
        catenane = cat_in
    else:
        catenane = cat_in.copy()
    catenane['at_type'] = 1
    catenane['bond_type'] = 1
    catenane['angle_type'] = 1
    return catenane
