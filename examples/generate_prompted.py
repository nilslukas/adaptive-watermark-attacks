import os

import numpy as np
import torch
import transformers
from torchvision.utils import save_image
from tqdm import tqdm

from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.generate_image_args import GenerateImageArgs
from src.arguments.model_args import ModelArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.datasets.prompt_dataset import PromptDataset
from src.models.generators.image_generator import ImageGenerator
from src.models.model_factory import ModelFactory
from src.utils.highlited_print import bcolors
from src.utils.utils import set_random_seed
from src.watermarking_key.wm_key_factory import WatermarkingKeyFactory


def parse_args():
    parser = transformers.HfArgumentParser((GenerateImageArgs,
                                            WatermarkingKeyArgs,
                                            ModelArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def generate_images(generate_image_args: GenerateImageArgs,
                    wm_key_args: WatermarkingKeyArgs,
                    model_args: ModelArgs,
                    env_args: EnvArgs,
                    config_args: ConfigArgs):
    """
    Generate images using a (optionally watermarked image generator)
    """
    if config_args.exists():
        generate_image_args = config_args.get_generate_image_args()
        wm_key_args = config_args.get_watermarking_key_args()
        model_args = config_args.get_model_args()
        env_args = config_args.get_env_args()

    # FOR SEEDING
    set_random_seed(env_args.seed)
    generator: ImageGenerator = ModelFactory.from_model_args(model_args, wm_key_args=wm_key_args,
                                                             env_args=env_args).load()
    wm_key = None
    print(wm_key_args)
    if wm_key_args.name is not None:  # Load the pre-trained watermarking key
        wm_key = WatermarkingKeyFactory.from_watermark_key_args(wm_key_args, env_args=env_args)
        if os.path.isfile(wm_key.wm_key_args.key_ckpt):
            wm_key = wm_key.load()
            print(f"> Loaded the pre-trained watermarking key {wm_key}.")
        else:
            dirname = os.path.dirname(wm_key.wm_key_args.key_ckpt)
            os.makedirs(dirname, exist_ok=True)
            wm_key = wm_key.keygen(generator=generator)
            print(f"> Generated and saved the watermarking key {wm_key}.")
        print(f"> Watermarking Message Set: {wm_key.get_message()}")

    generator.set_watermarking_key(wm_key)

    if hasattr(wm_key_args, "disable") and wm_key_args.disable:
        generator.set_watermarking_key(None)
        print("> DISABLING WATERMARK")

    if generate_image_args.dataset is not None:
        dataset = PromptDataset(generate_image_args.dataset)
    else:
        dataset = [""]

    # generate images
    num_prompts = min(len(dataset), generate_image_args.end_prompt - generate_image_args.start_prompt)
    num_images_per_prompt = generate_image_args.num_images
    num_images = num_images_per_prompt * num_prompts
    num_prompts_per_batch = max(int(np.floor(env_args.batch_size / num_images_per_prompt)), 1)
    num_batches_per_prompt = int(np.ceil(num_images_per_prompt / env_args.batch_size))
    num_images_single = num_images_per_prompt if num_batches_per_prompt == 1 else env_args.batch_size

    outdir = os.path.abspath(generate_image_args.outdir)
    print(f"> Generating {num_images} images to '{bcolors.OKGREEN}{outdir}{bcolors.ENDC}'.")
    os.makedirs(outdir, exist_ok=True)

    if generate_image_args.dump_prompts:
        prompt_dir = f"{outdir}_prompts"
        os.makedirs(prompt_dir, exist_ok=True)
        for i in range(num_images_per_prompt):
            dataset.save(prompt_dir, f"_{i}")

    done_prompts = generate_image_args.start_prompt
    for start_ix in tqdm(range(done_prompts, generate_image_args.end_prompt, num_prompts_per_batch),
                         desc="Write Images"):
        end_ix = min(start_ix + num_prompts_per_batch, len(dataset))
        prompts = dataset[start_ix:end_ix]
        x = []
        for ctr in range(num_batches_per_prompt):
            with torch.no_grad():
                _, _x = generator.generate(num_images=num_images_single, prompt=prompts)
                x.append(_x)
        x = torch.cat(x)
        for j, x_i in enumerate(x):
            image_prompt_ix = (j + 1) % num_images_per_prompt
            filename = os.path.join(outdir, f"image_{done_prompts}_{image_prompt_ix}.png")
            save_image(x_i.cpu(), filename)  # Save the image as PNG
            done_prompts += 1 if image_prompt_ix == 0 else 0


if __name__ == "__main__":
    generate_images(*parse_args())
