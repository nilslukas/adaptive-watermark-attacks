import json
import os

import numpy as np
import transformers
from tqdm import tqdm

from src.arguments.config_args import ConfigArgs
from src.arguments.detect_args import DetectArgs
from src.arguments.env_args import EnvArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.datasets.image_folder import ImageFolderDataset
from src.utils.highlited_print import bcolors
from src.utils.utils import set_random_seed, plot_images
from src.watermarking_key.wm_key_factory import WatermarkingKeyFactory


def parse_args():
    parser = transformers.HfArgumentParser((DetectArgs,
                                            WatermarkingKeyArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def detect_watermark(detect_args: DetectArgs,
                     wm_key_args: WatermarkingKeyArgs,
                     env_args: EnvArgs,
                     config_args: ConfigArgs):
    """
    Detects the presence of watermarks in a folder of images.

    Input:
    (1) Folder of images,
    (2) watermarking key checkpoint (*.pt file)

    Output:
    (1) The expected p-value
    (2) A list of all p-values
    (3) detection accuracy (if a threshold is defined).
    """
    if config_args.exists():
        detect_args = config_args.get_detect_args()
        wm_key_args = config_args.get_watermarking_key_args()
        env_args = config_args.get_env_args()

    set_random_seed(env_args.seed)

    # load the watermarking key.
    wm_key = WatermarkingKeyFactory.from_watermark_key_args(wm_key_args, env_args=env_args).load()

    # load the image dataset
    dataset = ImageFolderDataset(detect_args.image_folder)
    dataloader = env_args.make_data_loader(dataset, shuffle=False)

    # verify the message in each image.
    p_values = []
    for x, _ in tqdm(dataloader, desc="Detection"):
        p_values.extend(wm_key.verify(wm_key.extract(x), wm_key.get_message())['p_values'])
        plot_images(x[:4].detach())

    result_dict = {
        "expected_p_value": np.mean(p_values),
        "p_values": p_values,
        "accuracy": len([p for p in p_values if p < wm_key.p_value_threshold]) / len(p_values),
    }
    print(f"> Measured Accuracy: {bcolors.OKGREEN}{result_dict['accuracy']}{bcolors.ENDC}")

    out_path = os.path.join(detect_args.image_folder, "robustness.json")
    if os.path.isfile(out_path):
        with open(out_path, "r") as f:
            old_dict = json.load(f)
        for key in result_dict:
            old_dict[key] = result_dict[key]
        result_dict = old_dict
    with open(out_path, "w+") as f:
        json.dump(result_dict, f)
    print(f"> Accuracy: {result_dict['accuracy']:.3f} (expected p-value: {result_dict['expected_p_value']:.3f})")

if __name__ == "__main__":
    detect_watermark(*parse_args())
