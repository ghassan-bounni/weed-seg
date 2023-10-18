from configs.base import load_config
from utils.utils import parse_args
from logger import setup_logging
from train import train
from eval import test


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    model_config = config["model"]
    data_config = config["data"]

    logger = setup_logging(output=args.log)

    if args.mode == "train":
        train(
            model_config,
            data_config,
            args.checkpoint,
            args.device,
            logger,
            verbose=args.verbose,
        )
    elif args.mode == "eval":
        test(
            model_config,
            data_config,
            args.checkpoint,
            args.device,
            logger,
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"Task {args.task} not recognized.")
