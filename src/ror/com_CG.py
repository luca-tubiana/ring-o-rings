import pandas as pd

from ror.abstract_CG import CatCG


class CatComCG(CatCG):
    """
    Coarse-grain a trajectory based on the center of mass and normal of the rings.
    """

    def _get_vertices(self) -> pd.DataFrame:
        return self.coords.groupby([CatCG.conf_id, CatCG.ring_id])[CatCG.coord_lab].mean()
