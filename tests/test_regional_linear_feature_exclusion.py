from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/scripts/core_model/0_drainage_emissions_model.py"
)
SPEC = spec_from_file_location("drainage_model", MODULE_PATH)
drainage_model = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(drainage_model)


def test_exclude_regional_linear_feature_layers_drops_only_fixed_engert():
    download_dict = {
        "dadap": "s3://example/dadap.tif",
        "engert": "s3://example/engert.tif",
        "osm_roads": "s3://example/osm_roads.tif",
        "osm_canals": "s3://example/osm_canals.tif",
        "grip": "s3://example/grip.tif",
        "descals_type": "s3://example/descals_type.tif",
        "planted_forest_type": "s3://example/sdpt.tif",
    }

    filtered = drainage_model.exclude_regional_linear_feature_layers(download_dict)

    assert set(download_dict) - set(filtered) == {"engert"}
    assert "dadap" in filtered
    assert "descals_type" in filtered
    assert "osm_roads" in filtered
    assert "osm_canals" in filtered
    assert "grip" in filtered
