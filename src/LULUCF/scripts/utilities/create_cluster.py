"""
Run from src/LULUCF/
python -m scripts.utilities.create_cluster -n 1 -t 4 -cn AFOLU_flux_model_scripts
"""

import coiled
import argparse

from . import constants_and_names as cn

def create_cluster(cluster_name, n_workers, threads_per_worker, worker_memory, scheduler_memory, worker_cpu, idle_timeout):

    # Converts worker_memory from an integer to the required format (e.g., 8 to "8GiB")
    worker_memory_str = f"{worker_memory}GiB"
    scheduler_memory_str = f"{scheduler_memory}GiB"
    idle_timeout = f"{idle_timeout} minutes"

    cluster = coiled.Cluster(
        n_workers=n_workers,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout=idle_timeout,
        region="us-east-1",
        name=cluster_name,
        workspace='wri-forest-research',
        # mount_bucket="s3://gfw2-data",
        scheduler_memory = scheduler_memory_str,
        worker_memory = worker_memory_str,
        worker_cpu = worker_cpu,
        worker_options={
            "nthreads": threads_per_worker
        }
    )
    print(f"Cluster created with name: {cluster.name}")
    print(f"Number of workers: {n_workers}; worker memory: {worker_memory_str}; scheduler memory: {scheduler_memory_str}; threads per worker: {threads_per_worker}")
    return cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")
    parser.add_argument('-cn', '--cluster_name', type=str, help='Coiled cluster name')
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers for the cluster')
    parser.add_argument('-m', '--worker_memory', type=str, default='16', help='Memory per worker (default=16GiB)')
    parser.add_argument('-c', '--worker_cpu', type=str, default='2', help='Number of CPUs per worker (default=2 CPUs)')
    parser.add_argument('-t', '--threads_per_worker', type=int, default='2', help='Number of threads/worker (default=2)')
    parser.add_argument('-l', '--large_scale_mode', action='store_true', help='Use memory and workers for large-scale analysis')
    parser.add_argument('-i', '--idle_timeout', default=25, help='Timeout if idle is cluster (minutes)')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    n_workers = args.n_workers
    threads_per_worker = args.threads_per_worker
    worker_memory = args.worker_memory
    worker_cpu = args.worker_cpu
    idle_timeout = args.idle_timeout

    # Uses the larger workers if requested or if more workers are requested.
    # Assumes that if using more workers, you want bigger workers. 12 is a semi-arbitrary cutoff.
    if (args.large_scale_mode) or (n_workers > 12):

        worker_memory = '32'
        worker_cpu = 4
        idle_timeout = 15
        scheduler_memory = '64'

    else:
        scheduler_memory = worker_memory

    # Create the cluster with command line arguments
    create_cluster(cluster_name, n_workers, threads_per_worker, worker_memory, scheduler_memory, worker_cpu, idle_timeout)
