import numpy as np
import pandas as pd

import ror.ribbon as ribbon
from ror.abstract_CG import CatCG


class CatLocProp:
    """
    Compute and store local properties of a CG catenane
    """
    rib_norm_lab = ['rb_nx', 'rb_ny', 'rb_nz']

    def __init__(self, cgtraj):
        self._ribbon_normals = cgtraj.groupby(CatCG.conf_id).apply(CatLocProp.get_ribbon_normal)
        self.cgtraj = pd.concat([cgtraj, self._ribbon_normals], axis=1)
        self._twist = None
        self._bending = None
        self._bond_lengths = None
        self._norm_tang_coll = None
        self.cat_data = pd.concat(
            [self.cgtraj, self.norm_tang_coll, self.twist, self.bending, self.bond_lengths],
            axis=1)
        self.cat_data['collinearity'] = (self.cat_data['norm_tang_coll'] / self.cat_data['bond_lengths']) ** 2

    @property
    def norm_tang_coll(self):
        if self._norm_tang_coll is None:
            self._norm_tang_coll = self.cgtraj.groupby(CatCG.conf_id).apply(self.get_norm_tang_collinearity)
        return self._norm_tang_coll

    @property
    def twist(self):
        if self._twist is None:
            def df_twist(df):
                tw = ribbon.local_twist(df[CatLocProp.rib_norm_lab], df[CatCG.tan_lab])
                return pd.DataFrame(tw, columns=['twist'], index=df.index)

            self._twist = self.cgtraj.groupby(CatCG.conf_id).apply(df_twist)
        return self._twist

    @property
    def bending(self):
        if self._bending is None:
            def df_bending(df):
                bending = ribbon.bending_angles(df[CatCG.tan_lab])
                return pd.DataFrame(bending, columns=['bend'], index=df.index)

            self._bending = self.cgtraj.groupby(CatCG.conf_id).apply(df_bending)
        return self._bending

    @property
    def bond_lengths(self):
        if self._bond_lengths is None:
            def df_bl(df):
                bl = ribbon.bond_lengths(df[CatCG.tan_lab])
                return pd.DataFrame(bl, columns=['bond_lengths'], index=df.index)

            self._bond_lengths = self.cgtraj.groupby(CatCG.conf_id).apply(df_bl)
        return self._bond_lengths

    @staticmethod
    def get_norm_tang_collinearity(normtang: pd.DataFrame) -> pd.DataFrame:
        """
        Return amount of collinearity between ring normal and tangent
        Parameters
        ----------
        normtang: pd.DataFrame
            normal and tangent vectors

        Returns
        -------
        pd.DataFrame
            averaged amount of collinearity between n_i and t_i, t_(i-1) for ring i
        """
        tang = normtang[CatCG.tan_lab].values
        norm = normtang[CatCG.norm_lab].values
        cosines1 = np.einsum('ij,ij->i', norm, tang)
        # cosines2 = np.einsum('ij,ij->i', norm, np.roll(tang, 1, axis=0))
        # collinearity = 0.5 * (cosines1 + cosines2)
        collinearity = cosines1
        return pd.DataFrame(collinearity, columns=['norm_tang_coll'], index=normtang.index)

    @staticmethod
    def get_ribbon_normal(normtang: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the component of the ring normals orthogonal to the tangent
        Parameters
        ----------
        normtang: pd.DataFrame
            normals and tangents

        Returns
        -------
        pd.DataFrame
        """
        normals = normtang[CatCG.norm_lab].values
        tangents = normtang[CatCG.tan_lab].values
        t_norms = np.linalg.norm(tangents, axis=1)
        tangents /= t_norms[:, np.newaxis]
        nt = np.einsum('ij,ij->i', normals, tangents)
        normals -= nt[:, np.newaxis] * tangents
        n_norms = np.linalg.norm(normals, axis=1)
        normals /= n_norms[:, np.newaxis]
        return pd.DataFrame(normals, columns=CatLocProp.rib_norm_lab, index=normtang.index)


class CatGlobProp:
    """
    Compute and store global properties of a CG catenane
    """

    def __init__(self, loc_data):
        # assert isinstance(data, CatLocProp), "data must be of type ror.CatLocalProperties"
        self.cat_data = loc_data
        self._com = None
        self._rg2 = None
        self._twist = None
        self._writhe = None
        self._contour_length = None
        self._norm_corr = None
        self._bend_corr = None
        self._collinearity = None
        self.data = pd.concat(
            [self.rg2, self.twist, self.writhe, self.contour_length, self.bend_corr, self.norm_corr,
             self.collinearity], axis=1)

    @property
    def com(self):
        """
        Centers of masses of all conformations

        Returns
        -------
        pd.DataFrame
        """
        if self._com is None:
            self._com = self.cat_data.groupby(CatCG.conf_id)[CatCG.vert_lab].mean()
        return self._com

    @property
    def rg2(self):
        if self._rg2 is None:
            def df_rg2(df):
                return ribbon.rg2(df[CatCG.vert_lab])

            self._rg2 = self.cat_data.groupby(CatCG.conf_id)[CatCG.vert_lab].apply(df_rg2)
        return self._rg2.rename('rg2')

    # @property
    # def rg2(self) -> pd.DataFrame:
    #    if self._rg2 is None:
    #        rg2 = self.gyroeigvals[['ev_1', 'ev_2', 'ev_3']].sum(axis=1)
    #        self._rg2 = pd.DataFrame(rg2, columns=['rg2'], index=self.gyroeigvals.index)
    #    return self._rg2

    @property
    def twist(self):
        if self._twist is None:
            self._twist = self.cat_data.groupby(CatCG.conf_id)['twist'].sum()
        return self._twist

    @property
    def writhe(self):
        if self._writhe is None:
            def df_writhe(df):
                return ribbon.writhe(df[CatCG.vert_lab].values)

            self._writhe = self.cat_data.groupby(CatCG.conf_id).apply(df_writhe)
        return self._writhe.rename('writhe')

    @property
    def contour_length(self):
        """
        Length of the catenane obtained summing the length of all edges.

        Returns
        -------
        pd.DataFrame
        """
        if self._contour_length is None:
            self._contour_length = self.cat_data.groupby(CatCG.conf_id)['bond_lengths'].sum()  # sum over all mol_ids
        return self._contour_length.rename('contour_length')

    @property
    def norm_corr(self):
        """
        Correlations between normals

        Returns
        -------
        pd.DataFrame
        """
        if self._norm_corr is None:
            self._norm_corr = self.cat_data.groupby(CatCG.conf_id).apply(self.get_normal_corr)
        return self._norm_corr

    @property
    def collinearity(self):
        """
        Collinearity between normals and tangents
        Returns
        -------
        pd.DataFrame
        """
        if self._collinearity is None:
            self._collinearity = self.cat_data.groupby(CatCG.conf_id)['collinearity'].sum()
        return self._collinearity

    @property
    def bend_corr(self):
        """
        Correlations between tangents (angles)

        Returns
        -------
        pd.DataFrame
        """
        if self._bend_corr is None:
            self._bend_corr = self.cat_data.groupby(CatCG.conf_id).apply(self.get_bend_corr)
        return self._bend_corr

    @staticmethod
    def get_normal_corr(norm: pd.DataFrame) -> pd.DataFrame:
        """
        Compute  the normals' correlation function within a catenane.

        Parameters
        ----------
        norm: pd.DataFrame
            normals to a SINGLE catenane conformation.

        Returns
        -------
        pd.DataFrame
            average values of n*n_i, with i the shift index along the backbone
        """
        normals = norm[CatCG.norm_lab].values
        cov = normals @ normals.T
        scal_prods = np.abs(cov.copy())
        for i in range(1, len(normals)):
            scal_prods[i] = np.roll(scal_prods[i], -i)
        return pd.DataFrame(scal_prods[:, 1:], columns=[f'C_norm_{i}' for i in range(1, len(normals))]).mean()

    @staticmethod
    def get_bend_corr(norm: pd.DataFrame) -> pd.DataFrame:
        """
        Compute  the tangents' correlation function within a catenane.

        Parameters
        ----------
        norm: pd.DataFrame
            normals to a SINGLE catenane conformation.

        Returns
        -------
        pd.DataFrame
            average values of n*n_i, with i the shift index along the backbone
        """
        tangents = norm[CatCG.tan_lab].values
        cov = tangents @ tangents.T
        scal_prods = cov.copy()
        for i in range(1, len(tangents)):
            scal_prods[i] = np.roll(scal_prods[i], -i)
        return pd.DataFrame(scal_prods[:, 1:], columns=[f'C_tan_{i}' for i in range(1, len(tangents))]).mean()
