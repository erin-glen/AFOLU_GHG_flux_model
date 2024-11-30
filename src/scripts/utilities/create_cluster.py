# create_cluster.py

import coiled
import argparse

def create_cluster(n_workers, worker_vm_types=None, worker_memory=None, worker_cpu=None):
    """
    Creates a Coiled cluster with specified parameters.

    Args:
        n_workers (int): Number of workers for the cluster.
        worker_vm_types (list, optional): List of worker VM types (e.g., ['r7i.xlarge']).
        worker_memory (int, optional): Memory per worker in GiB.
        worker_cpu (int, optional): Number of CPUs per worker.
    """

    cluster_params = {
        'name': "drainage_cluster",
        'n_workers': n_workers,
        'use_best_zone': True,
        'compute_purchase_option': "spot_with_fallback",
        'idle_timeout': "15 minutes",
        'region': "us-east-1",
        'account': 'wri-forest-research',
    }

    if worker_vm_types:
        cluster_params['worker_vm_types'] = worker_vm_types
        print(f"Creating cluster with worker VM types: {worker_vm_types}")
    else:
        # Ensure that worker_memory and worker_cpu are provided
        if worker_memory is None or worker_cpu is None:
            raise ValueError("worker_memory and worker_cpu must be specified if worker_vm_types is not provided.")
        worker_memory_str = f"{worker_memory}GiB"
        cluster_params['worker_memory'] = worker_memory_str
        cluster_params['worker_cpu'] = worker_cpu
        print(f"Creating cluster with worker_memory: {worker_memory_str} and worker_cpu: {worker_cpu}")

    cluster = coiled.Cluster(**cluster_params)

    print(f"Cluster created with name: {cluster.name}")
    print(f"Number of workers: {n_workers}")
    return cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")

    parser.add_argument('-n', '--n_workers', type=int, default=1,
                        help='Number of workers for the cluster')

    parser.add_argument('--worker_vm_types', nargs='+', default=None,
                        help='List of worker VM types (e.g., r7i.xlarge)')

    parser.add_argument('-m', '--worker_memory', type=int, default=None,
                        help='Memory per worker in GiB (e.g., 8)')

    parser.add_argument('-c', '--worker_cpu', type=int, default=None,
                        help='Number of CPUs per worker (e.g., 4)')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    create_cluster(
        n_workers=args.n_workers,
        worker_vm_types=args.worker_vm_types,
        worker_memory=args.worker_memory,
        worker_cpu=args.worker_cpu
    )

"""
Example:

To create a cluster using r7i.xlarge with 100 workers:

python src/scripts/utilities/create_cluster.py -n 100 --worker_vm_types r7i.xlarge

"""

