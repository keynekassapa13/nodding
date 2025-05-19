import argparse

from nodding.video import process_video

def main():
    parser = argparse.ArgumentParser(description="Detect nodding in a video.")
    parser.add_argument("--input", required=True,
                        help="Path to the input video file.")
    parser.add_argument("--output", default="data/output/", 
                        help="Path to the output directory.")
    args = parser.parse_args()
    process_video(args.input, args.output)

if __name__ == "__main__":
    main()