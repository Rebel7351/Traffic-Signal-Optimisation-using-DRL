import os
import sys
import argparse
from datetime import datetime
import pandas

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Fixed Traffic Light Control"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="D:\\mtech\\projects\\Thesis Work\\4thSem\\TrafficSignalOptimization\\experiments\\real_traffic\\osm_nets\\new_delhi\\osm.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-gui", action="store_true", default=True, help="Run with visualization on SUMO.\n")
    prs.add_argument("-seconds", dest="seconds", type=int, default=3600, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-min_green", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-max_green", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")

    args = prs.parse_args()

    experiment_time = str(datetime.now()).split(".")[0].replace(":", "_")
    out_csv_fixed = f"outputs/real_{experiment_time}_ftlc"

    env = SumoEnvironment(
        net_file="D:\\mtech\\projects\\Thesis Work\\4thSem\\TrafficSignalOptimization\\experiments\\real_traffic\\osm_nets\\new_delhi\\osm.net.xml",
        route_file=args.route,
        out_csv_name=out_csv_fixed,
        use_gui=False,

        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        sumo_warnings=False,
        fixed_ts=True,  # Set fixed_ts to True for fixed traffic light control
    )

    for run in range(1, 100):  
        env.reset()
        done = {"__all__": False}
        step=0
        while not done["__all__"]:
            _, _, done, _ = env.step({})  # Take a step with no actions (fixed traffic light control)
            step+=1
            if(step>args.seconds):
                break
        env.save_csv(out_csv_fixed, run)
        env.close()