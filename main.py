import os
import dotenv
import wandb
from utils.utils import parse_args
from configs.base import load_config
from logger import setup_logging
from train import train
from eval import test


if __name__ == "__main__":
    args = parse_args()
    dotenv.load_dotenv()
    config = load_config(args.config)

    model_config = config["model"]

    setup_logging(name="StemDetectionLogger")

    if args.mode == "train":
        train_config = config["train"]
        val_config = config["eval"]

        wandb.login(key=os.environ["WANDB_API_KEY"])

        with wandb.init(
            project=os.environ.get("WANDB_PROJECT", "Weed-Seg"),
            config=config,
            name=os.environ.get("WANDB_NAME", None),
            notes=os.environ.get("WANDB_NOTES", None),
        ):
            train(
                model_config,
                train_config,
                val_config,
                args.data,
                int(args.save_interval),
                args.seed,
                args.checkpoint,
            )
    elif args.mode == "eval":
        test_config = config["eval"]
        test(
            model_config,
            test_config,
            args.data,
            args.checkpoint,
            args.output,
        )
    else:
        raise ValueError(f"Task {args.mode} not recognized.")
