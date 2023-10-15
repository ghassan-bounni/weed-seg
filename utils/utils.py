import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Model Pipeline")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config file."
    )
    parser.add_argument(
        "--mode",
        default="train",
        help="Mode to run the pipeline in.",
        choices=["train", "eval"],
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--output", default="output", help="Path to the output directory."
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run the pipeline on."
    )
    parser.add_argument("--seed", default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--verbose", default=False, help="Whether to print the logs or not."
    )
    args = parser.parse_args()
    return args


