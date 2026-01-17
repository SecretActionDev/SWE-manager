import argparse
from swe_manager.data_loader import (
    load_swebench_lite,
    get_instance_row,
)
from swe_manager.preprocessing import(process_data)
from swe_manager.clustering import(cluster_issue)
from swe_manager.agent_selector import(assign_agents)
import os

def main():
    parser = argparse.ArgumentParser(
        prog="swe-manager",
        description="SWE-Manager: Cluster-based agent selector for SWE-Bench"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- run command ----
    run_parser = subparsers.add_parser(
        "run",
        help="Run SWE-Manager on a SWE-Bench instance"
    )
    run_parser.add_argument(
        "--instance_id",
        type=str,
        required=True,
        help="SWE-Bench instance ID (e.g., pandas-dev__pandas-12345)"
    )

    args = parser.parse_args()

    if args.command == "run":
        handle_run(args)


def handle_run(args):
    option = args.instance_id.lower()

    ### 1. load swe bench lite: currently this is only one supported
    df = load_swebench_lite()

    #### if option is "all"
    if option == "all":
        processed_df = process_data(df)
        clustered_res = cluster_issue(processed_df)
        assigned_agents = assign_agents(clustered_res)

    # instance = get_instance_row(df, instance_id)
    
    # print(" - Extract features")
    # print(" - Run HDBSCAN")
    # print(" - Select best agent")

    #### if option is subset of instance


if __name__ == "__main__":
    main()
