from src.arguments.env_args import EnvArgs
from src.arguments.evasion_args import EvasionArgs
from src.evasion.black_box.adaptive_diffusion import AdaptiveDiffusionAttack
from src.evasion.black_box.advnoise import AdvNoiseAttack
from src.evasion.black_box.blurring import BlurringAttack
from src.evasion.black_box.cropping import CroppingAttack
from src.evasion.black_box.gaussian_noise import GaussianNoiseAttack
from src.evasion.black_box.image_jittering import ImageJitteringAttack
from src.evasion.black_box.jpeg_compression import JPEGCompressionAttack
from src.evasion.black_box.quantization import QuantizationAttack
from src.evasion.black_box.rotation import RotationAttack
from src.evasion.evasion_attack import EvasionAttack
from src.utils.highlited_print import bcolors


class AttackFactory:

    @staticmethod
    def from_evasion_args(evasion_args: EvasionArgs, env_args: EnvArgs = None) -> EvasionAttack:
        """
        Load an evasion attack.
        """
        print(f"> Loading {bcolors.OKBLUE}{evasion_args.attack_name.capitalize()}{bcolors.ENDC} attack.")
        return {
            "gaussian-noise": GaussianNoiseAttack,
            "blurring": BlurringAttack,
            "jpeg-compression": JPEGCompressionAttack,
            "rotation": RotationAttack,
            "quantization": QuantizationAttack,
            "image-jittering": ImageJitteringAttack,
            "cropping": CroppingAttack,
            ## adaptive attacks
            "advnoise": AdvNoiseAttack,
            "adaptive-diffusion": AdaptiveDiffusionAttack
        }[evasion_args.attack_name](evasion_args, env_args)
