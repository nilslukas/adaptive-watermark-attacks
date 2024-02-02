from dataclasses import dataclass, field


@dataclass
class GenerateImageArgs:
    CONFIG_KEY = "gen_img_args"
    """  Generate Image Arguments """

    generator_ckpt: str = field(default=None, metadata={
        "help": "the checkpoint to the generator. alternatively, specify model args directly"
    })

    outdir: str = field(default="../generated_images/your_folder", metadata={
        "help": "the directory to save the generated images."
    })

    num_images: int = field(default=1_000, metadata={
        "help": "number of images to generate"
    })

    start_prompt: int = field(default=0, metadata={
        "help": "prompt index to start generating"
    })

    end_prompt: int = field(default=1_000, metadata={
        "help": "prompt index to stop at"
    })

    dataset: str = field(default=None, metadata={
        "help": "the prompt dataset to use for generation"
    })

    dump_prompts: bool = field(default=False, metadata={
        "help": "whether or not to save the prompt dataset used for generation"
    })

    generation_seed: int = field(default=None, metadata={
        "help": "the seed to use for generation. None if a different seed should be used every time"
    })
