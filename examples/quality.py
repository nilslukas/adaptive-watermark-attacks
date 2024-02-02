import json
import os

import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.quality_args import QualityArgs
from src.metrics.metric_factory import MetricFactory
from src.utils.highlited_print import bcolors


def parse_args():
    parser = transformers.HfArgumentParser((QualityArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def measure_image_folder_quality(quality_args: QualityArgs,
                                 env_args: EnvArgs,
                                 config_args: ConfigArgs):
    """
    This script measures the quality given a path to an image folder dataset and returns a single
    number that denotes the quality (with whatever metric was specified).
    The image folder should be flat (i.e., the folder name should contain image files).

    Input:
    (1) A flat image folder dataset.

    Output:
    (1) A single number that denotes the quality of the dataset.
    """
    if config_args.exists():
        quality_args = config_args.get_quality_args()
        env_args = config_args.get_env_args()

    quality_metric = MetricFactory.from_quality_args(quality_args, env_args=env_args)
    q = quality_metric.measure()
    print(f"> Measured Quality: {bcolors.OKGREEN}{q}{bcolors.ENDC}")
    result_dict = {
        quality_args.metric: q
    }
    out_path = os.path.join(quality_args.image_folder, "robustness.json")
    if os.path.isfile(out_path):
        with open(out_path, "r") as f:
            old_dict = json.load(f)
        for key in result_dict:
            old_dict[key] = result_dict[key]
        result_dict = old_dict
    with open(out_path, "w+") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    measure_image_folder_quality(*parse_args())
