"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/
python -m src.utilities.create_cluster -n 1 -m 16 -cn LULUCF_model
python -m src.utilities.create_cluster -n 5 -m 32 -cn LULUCF_model
python -m src.utilities.create_cluster -n 20 -m 64 -cn LULUCF_model

Table of instance types: https://aws.amazon.com/ec2/instance-types/
Table of spot pricing: https://aws.amazon.com/ec2/spot/pricing/
These are the cheapest worker types and they have fewer vCPUs than usual for the memory.
This makes them less costly on AWS and use fewer Coiled credits.

List available worker types for Coiled clusters with: coiled.list_instance_types() in the Python shell

Using more than 1 thread/worker slows down processing a lot when there are more tasks than workers for the core LULUCF model,
which is the situation for large analyses, obviously.
"""

import coiled
import argparse
import sys


def create_cluster(cluster_name, n_workers, threads_per_worker, worker_memory):

    # Converts worker_memory from an integer to the required format (e.g., 8 to "8GiB")
    worker_memory_str = f"{worker_memory}GiB"
    scheduler_memory_str = f"{worker_memory}GiB"

    if worker_memory == 128:
        idle_timeout = 10
        scheduler_vm_type = "x2iedn.xlarge"
        worker_vm_type = "x2iedn.xlarge"

    elif worker_memory == 64:
        idle_timeout = 15
        scheduler_vm_type = "x2gd.xlarge"
        worker_vm_type = "x2gd.xlarge"

    elif worker_memory == 32:
        idle_timeout = 20
        scheduler_vm_type = "x8g.large"
        worker_vm_type = "x8g.large"

    elif worker_memory == 16:
        idle_timeout = 25
        scheduler_vm_type = "x2gd.medium"
        worker_vm_type = "x2gd.medium"

    elif worker_memory == 8:
        idle_timeout = 25
        scheduler_vm_type = "r8g.medium"
        worker_vm_type = "r8g.medium"

    elif worker_memory == 4:
        idle_timeout = 25
        scheduler_vm_type = "m8g.medium"
        worker_vm_type = "m8g.medium"

    # t2.small not available with Coiled. t3.small has 2 vCPUs, so it's not actually Coiled credit-effective.
    elif worker_memory == 2:
        idle_timeout = 25
        scheduler_vm_type = "t3.small"
        worker_vm_type = "t3.small"

    # # Couldn't get a cluster started that used 1GB workers using t3a.micro, t3.micro, or t4g.micro. Don't know why.
    # elif worker_memory == 1:
    #     idle_timeout = 25
    #     scheduler_vm_type = "t3a.micro"
    #     worker_vm_type = "t3a.micro"

    else:
        sys.exit('Memory argument not 2, 4, 8, 16, 32, 64, or 128 GB')

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
        tags = {"project": "AFOLU_flux_model"},
        scheduler_vm_types = scheduler_vm_type,
        worker_vm_types = worker_vm_type,
        worker_options={
            "nthreads": threads_per_worker
        },
    )

    print(f"Cluster created with name: {cluster.name}")
    print(f"Number of workers: {n_workers}; worker memory: {worker_memory_str}; scheduler memory: {scheduler_memory_str}; threads per worker: {threads_per_worker}")
    return cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")
    parser.add_argument('-cn', '--cluster_name', type=str, help='Coiled cluster name')
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers for the cluster')
    parser.add_argument('-m', '--worker_memory', type=int, help='Memory per worker')
    parser.add_argument('-t', '--threads_per_worker', type=int, default='1', help='Number of threads/worker (default=1)')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    n_workers = args.n_workers
    worker_memory = args.worker_memory
    threads_per_worker = args.threads_per_worker

    # Create the cluster with command line arguments
    create_cluster(cluster_name, n_workers, threads_per_worker, worker_memory)
