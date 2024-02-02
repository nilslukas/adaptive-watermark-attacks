from dataclasses import dataclass, field


@dataclass
class AdaptiveDiffuserArgs:
    CONFIG_KEY = "adaptive_diffuser_args"
    """  Adaptive Diffuser Arguments """
    save_path: str = field(default=None, metadata={
        "help": "the path to save the adaptive diffuser model."
    })

    img2img_ckpt: str = field(default="madebyollin/taesd", metadata={
        "help": "the checkpoint to the image2image model.",
        "choices": ["stabilityai/sdxl-vae", "madebyollin/taesd", "surrogate_vae"]
    })

    lr: float = field(default=1e-4, metadata={
        "help": "the learning rate."
    })

    lambda_bce: float = field(default=.001, metadata={
        "help": "the weight of the bce loss."
    })

    lambda_lpips: float = field(default=.1, metadata={
        "help": "the weight of the lpips loss."
    })

    lambda_mse: float = field(default=10, metadata={
        "help": "the weight of the mse loss."
    })

    resume_from: str = field(default=None, metadata={
        "help": "the path to the adaptive diffuser model."
    })

    opt: str = field(default="adam", metadata={
        "help": "the optimizer to use.",
        "choices": ["adam", "sgd"]
    })

    num_reversal_inference_steps: int = field(default=5, metadata={
        "help": "the number of steps to run the reversal inference."
    })