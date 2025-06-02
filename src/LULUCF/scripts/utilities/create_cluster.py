"""
Run from src/LULUCF/
python -m scripts.utilities.create_cluster -n 1 -t 2 -m 16 -cn LULUCF_model
python -m scripts.utilities.create_cluster -n 5 -t 3 -m 32 -cn LULUCF_model
python -m scripts.utilities.create_cluster -n 20 -t 4 -m 64 -cn LULUCF_model

Used x8g.large for 32 GB workers because using x2gd for large clusters kept giving me the error after the cluster was created
"error sending local AWS or Google Cloud credentials to cluster: Exception while trying to call remote method 'aws_update_credentials' using comm None."
Then the cluster would terminate.
"""

import coiled
import argparse
import sys

from . import constants_and_names as cn

def create_cluster(cluster_name, n_workers, threads_per_worker, worker_memory, idle_timeout):

    # Converts worker_memory from an integer to the required format (e.g., 8 to "8GiB")
    worker_memory_str = f"{worker_memory}GiB"
    scheduler_memory_str = f"{worker_memory}GiB"
    idle_timeout = f"{idle_timeout} minutes"

    # Uses the larger workers if requested or if more workers are requested.
    # Assumes that if using more workers, you want bigger workers. 12 is a semi-arbitrary cutoff.
    if worker_memory == 64:
        idle_timeout = 15
        scheduler_vm_type = "x2gd.xlarge"
        worker_vm_type = "x2gd.xlarge"

    elif worker_memory == 32:
        idle_timeout = 20
        scheduler_vm_type = "x8g.large"
        worker_vm_type = "x8g.large"

    elif worker_memory == 16:
        scheduler_vm_type = "x2gd.medium"
        worker_vm_type = "x2gd.medium"

    else:
        sys.exit('Memory argument not 16, 32, or 64 GB')

    cluster = coiled.Cluster(
        n_workers=n_workers,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout=idle_timeout,
        region="us-east-1",
        name=cluster_name,
        workspace='wri-forest-research',
        # mount_bucket="s3://gfw2-data",
        tags = {"project": "AFOLU_flux_model"},
        scheduler_vm_types = scheduler_vm_type,
        worker_vm_types = worker_vm_type,
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
    parser.add_argument('-m', '--worker_memory', type=int, help='Memory per worker')
    parser.add_argument('-t', '--threads_per_worker', type=int, default='2', help='Number of threads/worker (default=2)')
    parser.add_argument('-i', '--idle_timeout', default=25, help='Timeout if idle is cluster (minutes)')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    n_workers = args.n_workers
    worker_memory = args.worker_memory
    threads_per_worker = args.threads_per_worker
    idle_timeout = args.idle_timeout

    # Create the cluster with command line arguments
    create_cluster(cluster_name, n_workers, threads_per_worker, worker_memory, idle_timeout)
