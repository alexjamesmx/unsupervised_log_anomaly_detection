import os
import yaml
from logadempirical.helpers import arg_parser
from logging import getLogger

from accelerate import Accelerator
import logging
import argparse

from run_train import run_train
from run_load import run_load
# Logging config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

accelerator = Accelerator()


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    if args.config_file is not None and os.path.exists(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config_args = argparse.Namespace(**config)
            for k, v in config_args.__dict__.items():
                if v is not None:
                    setattr(args, k, v)
        print(f"Loaded config from {config_file}!")

    logger = getLogger(args.model_name)
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Main: Output directory: {args.output_dir}")
    if args.train and not args.load:
        run_train(args, accelerator,  logger)
    elif args.load and not args.train:
        run_load(args, accelerator, logger)
    else:
        raise ValueError("Either train or load must be True")
