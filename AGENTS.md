# Project Memory

## Coiled Cluster Launch

For this repo, launch Coiled clusters from WSL, not from the Windows conda environments. The expected local client/runtime is the WSL conda environment `coiled_20251119`.

Do not use the Windows `coiled_env` conda environment for cluster launch or Dask client work. It can attach with mismatched Python/Dask versions and cause distributed tasks to be cancelled.

### Standard Setup

Use the WSL prompt, or run the equivalent command through `wsl.exe`. The known working distro is `Ubuntu-24.04`.

```bash
cd /mnt/c/GIS/git/AFOLU_GHG_flux_model
source /home/eglen/anaconda3/etc/profile.d/conda.sh
conda activate coiled_20251119
python --version
python -c "import coiled, dask, distributed; print(coiled.__version__, dask.__version__, distributed.__version__)"
```

The local WSL environment should report Python 3.12 and Dask/Distributed 2025.x. The Coiled software environment visible in the `wri-forest-research` workspace is `afolu-env_coiled_20251119`. The shorthand `coiled_20251119` refers to the local WSL conda environment.

### Launch Command

Use `src.scripts.utilities.create_cluster` from the repo root:

```bash
python -m src.scripts.utilities.create_cluster \
  -n 20 \
  -m 64 \
  -t 1 \
  -cn <cluster_name> \
  --software afolu-env_coiled_20251119 \
  --spot-policy spot_with_fallback
```

Recommended naming pattern: include the workflow and date, for example `organic_soil_maps_20260512_wsl`.

### Worker Sizing

Agents have discretion to size the cluster to the task. Use the smallest cluster that is likely to finish the job reliably.

- Smoke tests and single-tile tests: `-n 1` to `-n 8`.
- Small batch jobs: `-n 8` to `-n 25`.
- Medium preprocessing or map jobs: `-n 25` to `-n 75`.
- Large global jobs: `-n 100` to `-n 150`.
- Very large tasks may go up to `-n 200`, but avoid doing this unless the task is clearly global, expensive, and ready to run.

Default memory should usually be `-m 64` with `-t 1`. Use `-m 32` only for light jobs. Use `--spot-policy spot_with_fallback` by default; use `--spot-policy on-demand` only when interruption risk is more costly than compute cost.

### Verify Cluster State

After launch, verify that Coiled sees the cluster as `ready`:

```bash
python - <<'PY'
import coiled
target = "<cluster_name>"
for cluster in coiled.list_clusters()[:40]:
    if cluster.get("name") == target:
        print(cluster.get("name"), cluster.get("current_state", {}).get("state"), cluster.get("id") or cluster.get("cluster_id"))
PY
```

If the cluster is not `ready`, wait and check again before starting a large job. Do not start a long global run against a cluster that the helper reports as missing or stopped.

### Running Jobs Against The Cluster

Run pipeline scripts from the same WSL `coiled_20251119` environment. Pass the cluster name with the script's `--cluster_name` or `-cn` argument, depending on that script.

Example:

```bash
python -m src.scripts.postprocessing.visualization.create_organic_soil_presence_map \
  --cluster_name <cluster_name> \
  --organic_soil_version 20260508 \
  --probability_date 20260508 \
  --fscore_metric f1 \
  --threshold_method per-biome
```

For long jobs, write logs to `C:/tmp/afolu/...` or `/mnt/c/tmp/afolu/...` so progress can be checked without attaching to the process.

### Cleanup

Terminate clusters when the task is done or blocked:

```bash
python -m src.scripts.utilities.terminate_cluster <cluster_name>
```
