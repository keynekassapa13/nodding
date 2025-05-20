import argparse
import json
from box import Box
from nodding.video import process_video

def main():
    parser = argparse.ArgumentParser(description="Detect nodding in a video.")
    parser.add_argument("--input", required=True,
                        help="Path to the input video file.")
    parser.add_argument("--config", default="config.json",
                        help="Path to the configuration file.")
    parser.add_argument("--output", default="data/output/", 
                        help="Path to the output directory.")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = Box(json.load(f), default_box=True, default_box_attr=None)

    # Override with CLI args
    cfg.INPUT = args.input
    if args.output:
        cfg.OUTPUT.PATH = args.output
    print(f"Configuration: {cfg}")

    process_video(cfg)

if __name__ == "__main__":
    main()