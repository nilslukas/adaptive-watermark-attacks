from dataclasses import dataclass, field


@dataclass
class ModelArgs:
    CONFIG_KEY = "model_args"
    """ This class contains all parameters for the diffusion model generator. """

    # The following fields denote the key names in a checkpoint file.
    MODEL_KEY = "diffusion_model"  # state-dict
    MODEL_ARGS_KEY = "diffusion_model_args"  # model args

    model_name: str = field(default="stable-diffusion-1", metadata={
        "help": "name of the generation algorithm to use",
        "choices": ["stable-diffusion-1", "stable-diffusion-2"]  # currently only stable diffusion2 is supported
    })

    model_ckpt: str = field(default="google/ddpm-cat-256", metadata={
        "help": "name of the model to use"
    })

    pre_trained: bool = field(default=True, metadata={
        "help": "whether to load pre-trained model weights (either from storage or from the hub). "
                "Setting this to False means we load randomly initialized models."
    })

    scheduler: str = field(default="dpm", metadata={
        "help": "name of the scheduler to use",
        "choices": ["plms", "ddim", "dpm"]
    })

    guidance_scale: float = field(default=7.5, metadata={
        "help": "scale of the guidance"
    })

    scheduler_timesteps: int = field(default=20, metadata={
        "help": "number of timesteps to use for the scheduler"
    })
