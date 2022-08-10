import numpy as np
import pandas as pd

from ror.abstract_CG import CatCG


def lammpsdump_reader(file, cols=('at_id', 'mol_id', 'type', 'x', 'y', 'z')):
    """
    Reads the LAMMPSDUMP files (LAMMPS Trajectories) and stores the results
    in a pandas DataFrame
    """
    ts = 'TIMESTEP'
    nbr = 'NUMBER'
    crd = 'ATOMS id'
    n_cols = len(cols)
    timesteps = []
    confs = []
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            if ts in line:
                timesteps.append(int(f.readline()))
            if nbr in line:
                n_atoms = int(f.readline())
                for i in range(4):
                    f.readline()
            if crd in line:
                coords = np.zeros((n_atoms, n_cols))
                for i in range(n_atoms):
                    line = f.readline().split()
                    coords[i] = np.array(line)
                confs.append(coords)
            line = f.readline()
    confs = np.array(confs)
    n_atoms = confs.shape[1]
    n_frames = confs.shape[0]
    confs = confs.reshape((confs.shape[0] * confs.shape[1], confs.shape[2]))
    data = pd.DataFrame(confs, columns=cols)
    data[['x', 'y', 'z']] = data[['x', 'y', 'z']].astype(float)
    data[['at_id', 'mol_id', 'type']] = data[['at_id', 'mol_id', 'type']].astype(int)
    # dt = timesteps[1] - timesteps[0]
    data['timestep'] = np.repeat(np.array(timesteps), n_atoms)
    data['frame'] = np.repeat(np.arange(n_frames), n_atoms)
    data.set_index(['frame', 'timestep'], inplace=True)
    return data[list(cols)]


def lammpsdump_writer(filename: str, data: pd.DataFrame, boxlen):
    """
    Write a lammpsdump file to be visualized in VMD
    """
    assert {'mol_id', 'x', 'y', 'z'} <= set(data.columns)
    ts_str = "ITEM: TIMESTEP\n{:10d}\n"
    n_atn_str = "ITEM:  NUMBER OF ATOMS\n{:10d}\n"
    boxlen_str = f"ITEM: BOX BOUNDS pp pp pp\n" + 3 * f"{-boxlen} {boxlen}\n"
    coord_str = "ITEM: ATOMS id mol type xu yu zu\n"
    at_str = "{:7d} 1 1 {:16.9f} {:16.9f} {:16.9f}\n"
    with open(filename, 'w') as fout:
        for ts, new in data.groupby(level='timestep'):
            data = new.droplevel('timestep').reset_index()
            # timestep = ts
            fout.write(ts_str.format(ts))
            fout.write(n_atn_str.format(data.shape[0]))
            fout.write(boxlen_str)
            fout.write(coord_str)
            data.reset_index()
            for index, row in data.iterrows():
                fout.write(at_str.format(int(row['mol_id']), row['x'], row['y'], row['z']))


def knt_writer(filename: str, data: pd.DataFrame, colx='x', coly='y', colz='z'):
    """
    Write a .knt file to be analyzed with entang
    """
    n_atn_str = "{:10d}\n"
    at_str = "{:16.9f} {:16.9f} {:16.9f}\n"
    with open(filename, 'w') as fout:
        for ts, new in data.groupby(level='timestep'):
            data = new.droplevel('timestep').reset_index()
            fout.write(n_atn_str.format(data.shape[0] + 1))
            data.reset_index()
            for index, row in data.iterrows():
                fout.write(at_str.format(row[colx], row[coly], row[colz]))
            row = data.iloc[0]
            fout.write(at_str.format(row[colx], row[coly], row[colz]))


def reconstruct_traj(filelist):
    """
    Collate a set of pandas traj
    files into a unique trajectory, sorting according to timestep and dropping duplicates.

    Parameters
    ----------
    filelist: list
        list of files in lammpstrj format to be read by ror.IO.lammpsdump_reader

    Returns
    -------

    """
    data = [lammpsdump_reader(f) for f in filelist]
    ts = [d.index[-1][1] for d in data]
    data_sorted = [d for _, d in sorted(zip(ts, data))]
    shifts = np.array([d.index[-1][0] for d in data_sorted])
    # shifts -= shifts[0]
    for d, s in zip(data_sorted[1:], shifts[:-1]):
        d.index.set_levels(d.index.levels[0] + s, level='frame', inplace=True)
    traj = pd.concat(data_sorted)
    traj.reset_index(level=1, inplace=True)
    # DO NOT drop duplicates on float values, only on integer/strings
    traj = traj.drop_duplicates(subset=['timestep', 'at_id'])
    traj.set_index(['timestep'], append=True, inplace=True)
    return traj


def to_vtf_file(data, M, n, colx, coly, colz, nx, ny, nz, filename):
    """
        Write a .vtf file to be opened with VMD
        Note: assumes that all frames have the same value of M
    """
    at_str = "{:d} {:16.9f} {:16.9f} {:16.9f}\n"
    s = 0.5 * n / np.pi
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
