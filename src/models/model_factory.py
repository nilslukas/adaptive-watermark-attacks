from src.arguments.adaptive_diffuser_args import AdaptiveDiffuserArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.models.autoencoders.stega import StegaStampEncoder, StegaStampDecoder
from src.models.generators.image_generator import ImageGenerator
from src.models.generators.sdw_stable_diffusion import SDWStableDiffusion
from src.models.generators.stable_diffusion import StableDiffusion
from src.models.generators.trw_stable_diffusion import TRWStableDiffusion
from src.models.generators.wdm_stable_diffusion import WDMStableDiffusion
from src.models.img2img.adaptive_diffuser import AdaptiveDiffuser
from src.utils.highlited_print import bcolors


class ModelFactory:

    @staticmethod
    def load_stega_autoencoder(resolution: int, bitlen: int, image_channels: int = 3, fingerprint_resolution: int = 16):
        return StegaStampEncoder(resolution=resolution, IMAGE_CHANNELS=image_channels, fingerprint_size=bitlen,
                                 fingerprint_resolution=fingerprint_resolution)

    @staticmethod
    def load_stega_decoder(bitlen: int, decoder_arch: str) -> StegaStampDecoder:
        assert decoder_arch in ["resnet18", "resnet50", "resnet101"]
        return StegaStampDecoder(fingerprint_size=bitlen, model_type=decoder_arch)

    @staticmethod
    def from_adaptive_diffuser_args(adaptive_diffuser_args: AdaptiveDiffuserArgs, env_args: EnvArgs) -> AdaptiveDiffuser:
        """
        Load an adaptive diffuser
        """
        return AdaptiveDiffuser(adaptive_diffuser_args, env_args)

    @staticmethod
    def from_model_args(model_args: ModelArgs, wm_key_args: WatermarkingKeyArgs = None,
                        env_args: EnvArgs = None) -> ImageGenerator:
        """
        Instantiate an image generator. If watermarking keys are provided, we load a specific subclass of
        the generator that implements the watermarking method.
        """
        suffix = f"{wm_key_args.name} watermarking key" if wm_key_args is not None else "vanilla model"
        print(f"> Loading {bcolors.OKBLUE}{model_args.model_name}{bcolors.ENDC} ({suffix})")
        if model_args.model_name.startswith("stable-diffusion"):
            if wm_key_args is None or wm_key_args.name is None:  # Load a vanilla model
                return StableDiffusion(model_args=model_args, env_args=env_args)
            elif wm_key_args.name == "sdw":
                return SDWStableDiffusion(model_args=model_args, env_args=env_args)  # no model change required
            elif wm_key_args.name == "trw":
                return TRWStableDiffusion(model_args=model_args, env_args=env_args)
            elif wm_key_args.name == "wdm":
                return WDMStableDiffusion(model_args=model_args, env_args=env_args)
            else:
                raise ValueError(f"Unknown generator {wm_key_args.name}")
        else:
            raise ValueError(f"Unknown model name {model_args.model_name}")
