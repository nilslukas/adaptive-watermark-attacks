from dataclasses import dataclass, field


@dataclass
class EvasionArgs:
    CONFIG_KEY = "evasion_args"
    """ Attack arguments """

    attack_name: str = field(default="gaussian-noise", metadata={
        "help": "the name of the attack."
                "gaussian-noise: adds random Gaussian noise to each image"
                "blurring: blurs an image with a Gaussian kernel"
                "jpeg-compression: compresses and uncompresses an image via JPEG compression alg"
                "quantization: quantizes the colors of the image into discrete levels"
                "adaptive-noise: add adaptive noise against a surrogate classifier",
        "choices": ["gaussian-noise",
                    "blurring",
                    "jpeg-compression",
                    "rotation",
                    "quantization",
                    "cropping",
                    "image-jittering"
                    ## adaptive attacks
                    "advnoise"
                    "adaptive-diffusion"]
    })

    root_folder: str = field(default=None, metadata={
        "help": "the root folder containing the images to attack."
    })

    output_folder: str = field(default=None, metadata={
        "help": "the output folder to save the attacked images."
    })

    @staticmethod
    def load_specific_class(**kwargs):
        return {
            "plain": EvasionArgs,
            "blurring": BlurringEvasionArgs,
            "jpeg-compression": JpegCompressionEvasionArgs,
            "rotation": RotationEvasionArgs,
            "quantization": QuantizationEvasionArgs,
            "gaussian-noise": GaussianNoiseEvasionArgs,
            "image-jittering": ImageJitteringEvasionArgs,
            "cropping": CroppingEvasionArgs,
            ## adaptive attacks
            "advnoise": AdvNoiseEvasionArgs,
            "adaptive-diffusion": AdaptiveDiffusionEvasionArgs
        }[kwargs.setdefault("attack_name", "plain")]

    max_samples: int = field(default=None, metadata={
        "help": "limit maximum numbers of samples to process. None means all samples are processed."
    })

    verbose_attack: bool = field(default=True, metadata={
        "help": "whether the attack should be verbose"
    })


@dataclass
class BlurringEvasionArgs(EvasionArgs):
    """
    Blurring of an image.
    """
    blur_radius: float = field(default=2.0, metadata={
        "help": "blur radius. Larger makes a worse quality image."
    })


@dataclass
class QuantizationEvasionArgs(EvasionArgs):
    """
    Quantization of an image.
    """
    num_levels: int = field(default=16, metadata={
        "help": "the number fo levels to use for coloring"
    })


@dataclass
class JpegCompressionEvasionArgs(EvasionArgs):
    """
    JPEG compression of an image.
    """
    jpeg_quality: int = field(default=80, metadata={
        "help": "Amount of compression applied. Lower is worse quality"
    })


@dataclass
class GaussianNoiseEvasionArgs(EvasionArgs):
    """
    Gaussian noise on top of an image.
    """
    gaussian_noise_scale: float = field(default=0.01, metadata={
        "help": "scale of the Gaussian noise to apply to the image."
    })


@dataclass
class RotationEvasionArgs(EvasionArgs):
    """
    Rotation of an image.
    """
    rot_degree: int = field(default=60, metadata={
        "help": "rotation degree."
    })


@dataclass
class CroppingEvasionArgs(EvasionArgs):
    """
    Cropping of an image.
    """
    crop_scale: float = field(default=0.75, metadata={
        "help": "Cropping scale."
    })
    crop_ratio: float = field(default=0.75, metadata={
        "help": "Cropping ratio."
    })


@dataclass
class ImageJitteringEvasionArgs(EvasionArgs):
    """
    Jittering of an image.
    """
    jitter_strength: float = field(default=6, metadata={
        "help": "jitter strength."
    })


@dataclass
class AdvNoiseEvasionArgs(EvasionArgs):
    """
    Adversarial noise against the surrogate key.
    """
    surr_key_ckpt: str = field(default=None, metadata={
        "help": "the checkpoint to the surrogate watermarking key."
    })

    adaptive_noise_epsilon: float = field(default=4 / 255, metadata={
        "help": "the epsilon value for the adversarial noise."
    })

    adaptive_noise_lr: float = field(default=0.01, metadata={
        "help": "the learning rate for the adversarial noise."
    })

    adaptive_noise_opt_steps: int = field(default=5, metadata={
        "help": "the number of optimization steps for the adversarial noise."
    })


@dataclass
class AdaptiveDiffusionEvasionArgs(EvasionArgs):
    """
    Adversarial refining (adaptive attack)
    """
    surr_key_ckpt: str = field(default=None, metadata={
        "help": "the checkpoint to the surrogate watermarking key."
    })

    surr_diffusion_ckpt: str = field(default=None, metadata={
        "help": "the surrogate diffusion model checkpoint"
    })

    diffusion_magnitude: int = field(default=3, metadata={
        "help": "the magnitude of the diffusion. How many times to autoencode the image."
    })
