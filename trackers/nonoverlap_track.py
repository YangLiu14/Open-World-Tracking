#!/usr/bin/env python3
import argparse
import glob
import multiprocessing as mp
import os
import yaml

from online_tracker import OnlineTracker, OnlineGhostTracker
from offline_tracker import OfflineTracker
from utils import print_config


def nonoverlap_tracking(args):
    with open(args.config, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    print_config(config)

    props_dir = config["data"]["props_dir"]
    tracker = None
    if args.tracker == "online":
        tracker = OnlineTracker(config)
    elif args.tracker == "online-ghost":
        tracker = OnlineGhostTracker(config)
    elif args.tracker == "offline":
        tracker = OfflineTracker(config)

    print(f"Running {args.tracker} tracker.")
    for datasrc in args.datasrcs:
        print(f"Processing Videos in {datasrc}")
        videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(props_dir, datasrc, '*')))]
        for vid, vname in enumerate(videos[:10]):
            print(f"Processing video {vid + 1}/{len(videos)}: {vname}")
            tracker.run(datasrc, vname)

    print("Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script for Non-Overlap, then Track')
    parser.add_argument('--tracker', choices=["online", "online-ghost", "offline"])
    parser.add_argument('--config', default="./configs/example_tracker.yaml", type=str)
    parser.add_argument('--datasrcs', nargs='+', default=["ArgoVerse"], help="List of sources of dataset.")
    args = parser.parse_args()

    nonoverlap_tracking(args)


