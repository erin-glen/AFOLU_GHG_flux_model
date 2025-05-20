import numpy as np
import pytest
from affine import Affine

pytest.importorskip("geopandas")
pytest.importorskip("dask_geopandas")

import dask_geopandas as dgpd

from src.scripts.preprocessing.roads_canals import roads_canals_chunks_rio_update as rc


def test_create_fishnet_from_masked():
    masked = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    transform = Affine.identity()

    fishnet = rc.create_fishnet_from_masked(masked, transform)

    assert isinstance(fishnet, dgpd.GeoDataFrame)
    assert len(fishnet) == 2
    assert str(fishnet.crs) == "EPSG:3395"