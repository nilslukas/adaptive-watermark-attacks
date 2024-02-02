import torch

from src.arguments.env_args import EnvArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.utils.highlited_print import bcolors
from src.utils.web import download_and_unzip
from src.watermarking_key.ptw_wm_key import PTWWatermarkingKey
from src.watermarking_key.sdw_wm_key import SDWWatermarkingKey
# from src.watermarking_key.ptw_wm_key import PTWWatermarkingKey
from src.watermarking_key.trw_wm_key import TRWWatermarkingKey
from src.watermarking_key.wdm_wm_key import WDMWatermarkingKey
from src.watermarking_key.wm_key import WatermarkingKey


class WatermarkingKeyFactory:
    """
    Instantiate watermarking keys.
    """

    @staticmethod
    def from_checkpoint(ckpt: str, env_args: EnvArgs = None) -> WatermarkingKey:
        ckpt = download_and_unzip(ckpt)
        print(ckpt)
        watermark_args = torch.load(ckpt)[WatermarkingKeyArgs.WM_KEY_ARGS_KEY]
        return WatermarkingKeyFactory.from_watermark_key_args(watermark_args, env_args=env_args)

    @staticmethod
    def from_watermark_key_args(watermark_args: WatermarkingKeyArgs, env_args: EnvArgs = None) -> WatermarkingKey:
        """
        Loads a watermarking key.
        """
        print(f"> Loading a watermarking key of type '{bcolors.OKBLUE}{watermark_args.name}{bcolors.ENDC}'")
        if watermark_args.name == "ptw":
            wm_key = PTWWatermarkingKey(watermark_args, env_args=env_args)
        elif watermark_args.name == "wdm":
            wm_key = WDMWatermarkingKey(watermark_args, env_args=env_args)
        elif watermark_args.name == "trw":
            wm_key = TRWWatermarkingKey(watermark_args, env_args=env_args)
        elif watermark_args.name == "sdw":
            wm_key = SDWWatermarkingKey(watermark_args, env_args=env_args)
        else:
            raise ValueError(watermark_args.name)

        return wm_key
