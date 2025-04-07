#!/usr/bin/env python3
"""
create_cluster.py

A merged Coiled cluster creation script for organic_soils that supports:
 - worker_vm_types for user-specified VM sizes (old approach),
 - numeric memory/CPU with optional large-scale overrides (new approach),
 - threads_per_worker and idle_timeout,
 - a default cluster name or user override.

Examples:

1) VM type approach (old style):
   python create_cluster.py -n 100 --worker_vm_types r7i.xlarge

2) Numeric memory & CPU approach (like LULUCF):
   python create_cluster.py -n 2 -m 16 -c 4 -t 2 -i 25

3) Large scale mode:
   python create_cluster.py -n 50 --large_scale_mode

"""
import coiled
import argparse


def create_cluster(
    cluster_name="drainage_cluster",
    n_workers=1,
    worker_vm_types=None,
    worker_memory="16",   # in GiB, as a string
    worker_cpu="2",       # as a string for CLI consistency
    threads_per_worker="2",
    idle_timeout=25,      # in minutes
    large_scale_mode=False
):
    """
    Creates a Coiled cluster with specified parameters.

    If 'worker_vm_types' is provided, the script uses that and ignores numeric memory/CPU.
    Otherwise it sets memory/CPU from 'worker_memory'/'worker_cpu', and can also handle
    large-scale overrides if 'large_scale_mode' is True or if n_workers > 12.

    Args:
        cluster_name (str): Name for the Coiled cluster.
        n_workers (int): Number of workers.
        worker_vm_types (list[str]): Optional list of worker VM types (e.g. ['r7i.xlarge']).
        worker_memory (str): Worker memory in GiB (string form, e.g. '16').
        worker_cpu (str): CPUs per worker (string form, e.g. '4').
        threads_per_worker (str): Threads per worker (e.g. '2').
        idle_timeout (int): Idle time in minutes before cluster auto-shutdown.
        large_scale_mode (bool): If True (or if n_workers > 12), switch to bigger default memory/CPUs.

    Returns:
        coiled.Cluster: The created Coiled cluster object.
    """

    # Convert user inputs to correct numeric types where needed
    # if they came in as strings from CLI
    try:
        worker_cpu_num = int(worker_cpu)
        threads_num = int(threads_per_worker)
        worker_memory_num = int(worker_memory)
    except ValueError:
        raise ValueError("Please provide numeric values for worker_memory, worker_cpu, and threads_per_worker.")

    # If user requested large-scale or the worker count is big, we override memory, CPU, idle timeout
    # just like LULUCF script does
    if large_scale_mode or n_workers > 12:
        worker_memory_num = 32
        worker_cpu_num = 4
        idle_timeout = 15
        scheduler_memory_num = 64
    else:
        # Use worker_memory as both worker & scheduler memory if we haven't set them bigger
        scheduler_memory_num = worker_memory_num

    # Prepare idle_timeout as a string with ' minutes'
    idle_timeout_str = f"{idle_timeout} minutes"

    # Prepare the cluster parameters dictionary
    cluster_params = {
        "n_workers": n_workers,
        "use_best_zone": True,
        "compute_purchase_option": "spot_with_fallback",
        "idle_timeout": idle_timeout_str,
        "region": "us-east-1",
        "name": cluster_name,
        "account": "wri-forest-research",
    }

    # If user gave a worker_vm_types list, use that approach. Else numeric memory/CPU approach.
    if worker_vm_types:
        cluster_params["worker_vm_types"] = worker_vm_types
        print(f"Creating cluster '{cluster_name}' with {n_workers} workers using VM types: {worker_vm_types}")
    else:
        # We convert memory to e.g. "16GiB" strings
        worker_memory_str = f"{worker_memory_num}GiB"
        scheduler_memory_str = f"{scheduler_memory_num}GiB"

        cluster_params["worker_memory"] = worker_memory_str
        cluster_params["scheduler_memory"] = scheduler_memory_str
        cluster_params["worker_cpu"] = worker_cpu_num
        cluster_params["worker_options"] = {"nthreads": threads_num}

        print(f"Creating cluster '{cluster_name}' with {n_workers} workers, "
              f"memory={worker_memory_str}, scheduler_memory={scheduler_memory_str}, "
              f"CPUs/worker={worker_cpu_num}, threads={threads_num}, idle_timeout={idle_timeout_str}")

    # Create the cluster
    cluster = coiled.Cluster(**cluster_params)
    print(f"Cluster created with name: {cluster.name}")
    print(f"Number of workers: {n_workers}")
    return cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with flexible parameters.")
    parser.add_argument("-cn", "--cluster_name", type=str, default="drainage_cluster", help="Coiled cluster name")
    parser.add_argument("-n", "--n_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--worker_vm_types", nargs="+", default=None, help="List of VM types (e.g. r7i.xlarge)")
    parser.add_argument("-m", "--worker_memory", type=str, default="16", help="Memory per worker in GiB (default=16)")
    parser.add_argument("-c", "--worker_cpu", type=str, default="2", help="CPUs per worker (default=2)")
    parser.add_argument("-t", "--threads_per_worker", type=str, default="2", help="Threads per worker (default=2)")
    parser.add_argument("-i", "--idle_timeout", type=int, default=25, help="Idle timeout in minutes (default=25)")
    parser.add_argument("-l", "--large_scale_mode", action="store_true", help="Use memory/workers for large-scale analysis")

    args = parser.parse_args()

    # Create the cluster
    create_cluster(
        cluster_name=args.cluster_name,
        n_workers=args.n_workers,
        worker_vm_types=args.worker_vm_types,
        worker_memory=args.worker_memory,
        worker_cpu=args.worker_cpu,
        threads_per_worker=args.threads_per_worker,
        idle_timeout=args.idle_timeout,
        large_scale_mode=args.large_scale_mode
    )
