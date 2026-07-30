"""Run a tiny, local-only end-to-end test of the portable map generator."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds


LAYER_COUNT = 8
WIDTH = 8
HEIGHT = 6


def main() -> None:
    bundle_dir = Path(__file__).resolve().parent
    generator = bundle_dir / "create_input_dataset_maps.py"

    with tempfile.TemporaryDirectory(prefix="afolu_map_smoke_") as temp_name:
        work_dir = Path(temp_name)
        input_dir = work_dir / "inputs"
        output_dir = work_dir / "outputs"
        input_dir.mkdir()

        datasets = []
        source_transform = from_bounds(0, 0, 4, 3, 4, 3)
        for index in range(1, LAYER_COUNT + 1):
            source = input_dir / f"layer_{index}.tif"
            values = np.arange(12, dtype="float32").reshape(3, 4) + index
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=4,
                height=3,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=source_transform,
                nodata=-9999,
            ) as raster:
                raster.write(values, 1)
            datasets.append(
                {
                    "name": f"smoke_layer_{index}",
                    "source": str(source),
                    "kind": "continuous",
                    "resampling": "bilinear",
                    "style": {
                        "cmap": "viridis",
                        "vmin": 1,
                        "vmax": 20,
                        "nodata_color": "#00000000",
                    },
                }
            )

        config = {
            "description": "Local-only portable bundle smoke test",
            "target": {
                "crs": "EPSG:4326",
                "bounds": [0, 0, 4, 3],
                "width": WIDTH,
                "height": HEIGHT,
            },
            "prefix_output_order": True,
            "write_aligned_geotiff": True,
            "datasets": datasets,
        }
        config_path = work_dir / "smoke.config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                str(generator),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "Portable smoke test failed.\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        manifest_path = output_dir / "input_dataset_maps.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pngs = sorted(output_dir.glob("*.png"))
        tiffs = sorted((output_dir / "aligned").glob("*.tif"))
        if len(manifest["layers"]) != LAYER_COUNT:
            raise AssertionError("Manifest did not contain all eight smoke-test layers")
        if len(pngs) != LAYER_COUNT or len(tiffs) != LAYER_COUNT:
            raise AssertionError("Smoke test did not produce eight PNGs and eight GeoTIFFs")
        for png in pngs:
            with Image.open(png) as image:
                if image.size != (WIDTH, HEIGHT) or image.mode != "RGBA":
                    raise AssertionError(f"Unexpected PNG properties for {png.name}")

    print("PASS: standalone generator produced 8 aligned PNGs and 8 GeoTIFFs locally.")


if __name__ == "__main__":
    main()

