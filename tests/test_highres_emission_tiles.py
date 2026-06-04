import unittest

from src.scripts.postprocessing.visualization import create_highres_emission_tiles as het


class HighresEmissionTilesTest(unittest.TestCase):
    def test_split_cli_items_accepts_commas_and_spaces(self) -> None:
        self.assertEqual(
            het._split_cli_items(["00N_100E,10N_100E", "20N_100E"]),
            ["00N_100E", "10N_100E", "20N_100E"],
        )

    def test_validate_data_types_rejects_non_emission_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports drained and burned"):
            het._validate_data_types(["combined_state"])

    def test_tile_output_uri_defaults_to_versioned_tile_tree(self) -> None:
        uri = het._tile_output_uri(
            outputs_base="s3://bucket/root/version_1_0_1",
            output_prefix=None,
            res_label="0_001deg",
            dataset="drained_total_Mg_CO2e_pixel_yr",
            run_name="run",
            interval="2021_2024",
            tile_id="00N_100E",
        )

        self.assertEqual(
            uri,
            "s3://bucket/root/version_1_0_1/0_001deg_tile_aggregation/"
            "drained_total_Mg_CO2e_pixel_yr/run/2021_2024/"
            "00N_100E__0_001deg__drained_total_Mg_CO2e_pixel_yr__2021_2024.tif",
        )

    def test_tile_output_uri_honors_output_prefix(self) -> None:
        uri = het._tile_output_uri(
            outputs_base="s3://bucket/root/version_1_0_1",
            output_prefix="s3://bucket/custom",
            res_label="0_001deg",
            dataset="burned_total_Mg_CO2e_pixel_yr",
            run_name="run",
            interval="2021_2024",
            tile_id="00N_100E",
        )

        self.assertEqual(
            uri,
            "s3://bucket/custom/burned_total_Mg_CO2e_pixel_yr/run/2021_2024/"
            "00N_100E__0_001deg__burned_total_Mg_CO2e_pixel_yr__2021_2024.tif",
        )


if __name__ == "__main__":
    unittest.main()
