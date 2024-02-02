import torch
import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.watermarking_key.wm_key import WatermarkingKey
from src.watermarking_key.wm_key_factory import WatermarkingKeyFactory


def parse_args():
    parser = transformers.HfArgumentParser((WatermarkingKeyArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def set_message(wm_key_args: WatermarkingKeyArgs,
                env_args: EnvArgs,
                config_args: ConfigArgs):
    """
    Sets a random message to a pre-trained watermarking key checkpoint
    """
    if config_args.exists():
        wm_key_args = config_args.get_watermarking_key_args()
        env_args = config_args.get_env_args()
        env_args = config_args.get_env_args()

    # FOR SEEDING
    print(wm_key_args)

    wm_key: WatermarkingKey = WatermarkingKeyFactory.from_watermark_key_args(wm_key_args, env_args=env_args)
    wm_key.load(torch.load(wm_key_args.key_ckpt, map_location='cpu'))

    wm_key.set_message(wm_key.sample_message(1))
    print(f"Set message: {wm_key.get_message()}")

    wm_key.save(wm_key_args.key_ckpt)


def check_message(
                  wm_key_args: WatermarkingKeyArgs,
                  env_args: EnvArgs,
                  config_args: ConfigArgs):
    """
    Generate images using a (optionally watermarked image generator)
    """
    if config_args.exists():
        wm_key_args = config_args.get_watermarking_key_args()
        env_args = config_args.get_env_args()

    wm_key: WatermarkingKey = WatermarkingKeyFactory.from_watermark_key_args(wm_key_args, env_args=env_args)
    wm_key.load(torch.load(wm_key_args.key_ckpt, map_location='cpu'))

    print(wm_key.wm_key_args.message)


if __name__ == "__main__":
    check_message(*parse_args())
    set_message(*parse_args())
