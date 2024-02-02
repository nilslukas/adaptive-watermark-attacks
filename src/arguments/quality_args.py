from dataclasses import dataclass, field


@dataclass
class QualityArgs:
    CONFIG_KEY = "quality_args"
    """  
    Evaluate the quality of a set of generated images.  
    """

    metric: str = field(default="fid", metadata={
        "help": "the checkpoint to the watermarking key.",
        "choices": ["fid", "clipscore"]
    })

    fid_features: int = field(default=2048, metadata={
        "help": "the number of features to use for FID."
                "Default is 2048, as used by https://github.com/mseitzer/pytorch-fid",
        "choices": [64, 192, 768, 2048]
    })

    image_folder: str = field(default=None, metadata={
        "help": "the folder containing the images to detect."
    })

    prompts_folder: str = field(default=None, metadata={
        "help": "the prompt dataset used for generation"
    })

    clip_model: str = field(default="openai/clip-vit-base-patch16", metadata={
        "help": "model used for clip score computation"
    })

    lpips_net: str = field(default="alex", metadata={
        "help": "model used for lpips score computation"
    })

    reference_image_folder: str = field(default=None, metadata={
        "help": "the folder containing the reference images."
    })
