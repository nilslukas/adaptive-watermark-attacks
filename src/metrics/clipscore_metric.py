import os

import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm

from src.metrics.base_metric import BaseMetric


class CLIPScoreMetric(BaseMetric):
    """
    Needs 'pip install torchmetrics[image]'
    """

    def measure(self) -> float:
        """
        Measure the CLIP score.
        """
        assert self.quality_args.image_folder is not None, "Clip needs to specify a folder with generated images"
        assert self.quality_args.prompts_folder is not None, "Clip needs to specify a path to a prompt dataset"
        metric = CLIPScore(model_name_or_path=self.quality_args.clip_model)
        metric.to(self.env_args.device)

        prompt_dataset = {}
        for prompt_file in os.listdir(self.quality_args.prompts_folder):
            stripped_name = os.path.basename(prompt_file).split(".")[0]
            path = os.path.join(self.quality_args.prompts_folder, prompt_file)
            with open(path, "r") as f:
                prompt_dataset[stripped_name] = "\n".join(f.readlines())

        image_loader = self.get_image_loader(self.quality_args.image_folder)

        for imgs, paths in tqdm(image_loader, desc=f"Calculating CLIP"):
            imgs_uint8 = (imgs * 255).to(torch.uint8).to(self.env_args.device)
            files = [os.path.basename(p).split(".")[0] for p in paths]
            prompts = [prompt_dataset[file] for file in files]
            metric.update(images=imgs_uint8, text=prompts)

        return metric.compute().item()
