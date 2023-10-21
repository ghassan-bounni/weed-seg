from utils.utils import parse_args
from configs.base import load_config
import logging
from logger import setup_logging
from train import train
from eval import test


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    model_config = config["model"]
    data_config = config["data"]

    logger = logging.Logger(name="StemDetectionLogger")

    setup_logging(name="StemDetectionLogger", output=args.output)

    if args.mode == "train":
        train_config = config["train"]
        train(model_config, data_config, train_config, args.checkpoint)
    elif args.mode == "eval":
        test_config = config["test"]
        test(model_config, test_config, args.checkpoint)
    else:
        raise ValueError(f"Task {args.task} not recognized.")
