"""
Run from src/LULUCF/
python -m scripts.utilities.create_cluster -n 1 -m 8
"""

import coiled
import argparse

from . import constants_and_names as cn

def create_cluster(n_workers, threads_per_worker, worker_memory, worker_cpu):

    # Convert worker_memory from an integer to the required format (e.g., 8 to "8GiB")
    worker_memory_str = f"{worker_memory}GiB"

    cluster = coiled.Cluster(
        n_workers=n_workers,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout="15 minutes",
        region="us-east-1",
        name="AFOLU_flux_model_scripts",
        workspace='wri-forest-research',
        # mount_bucket="s3://gfw2-data",
        worker_memory = worker_memory_str,
        worker_cpu = worker_cpu,
        worker_options={
            "nthreads": threads_per_worker
        }
    )
    print(f"Cluster created with name: {cluster.name}")
    print(f"Number of workers: {n_workers}; Worker memory: {worker_memory_str}; Threads per worker: {threads_per_worker}")
    return cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers for the cluster')
    parser.add_argument('-m', '--worker_memory', type=str, default='16', help='Memory per worker (default=16GiB)')
    parser.add_argument('-c', '--worker_cpu', type=str, default='2', help='Number of CPUs per worker (default=2 CPUs)')
    parser.add_argument('-t', '--threads_per_worker', type=int, default='3', help='Number of threads/worker (default=3)')
    parser.add_argument('-l', '--large_scale_mode', action='store_true', help='Use memory and workers for large-scale analysis')

    args = parser.parse_args()

    n_workers = args.n_workers
    threads_per_worker = args.threads_per_worker
    worker_memory = args.worker_memory
    worker_cpu = args.worker_cpu

    if args.large_scale_mode:

        worker_memory = '32'
        worker_cpu = 4

    # Create the cluster with command line arguments
    create_cluster(n_workers, threads_per_worker, worker_memory, worker_cpu)
