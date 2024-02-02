import os.path

import lpips
import torch
from tqdm import tqdm

from src.metrics.base_metric import BaseMetric


class LPIPSMetric(BaseMetric):

    def measure(self) -> float:
        """
        Measure the LPIPs score.
        """
        assert self.quality_args.image_folder is not None, "LPIPS needs to specify a folder with generated images"
        assert self.quality_args.reference_image_folder is not None, "LPIPS needs to specify a folder with reference images"

        metric = lpips.LPIPS(net=self.quality_args.lpips_net).to(self.env_args.device)

        image_loader = self.get_image_loader(self.quality_args.image_folder)
        source_loader = self.get_image_loader(self.quality_args.reference_image_folder)
        qualities = []
        for (image_batch, image_paths), (source_batch, source_paths) in tqdm(zip(image_loader, source_loader)):
            image_paths = [os.path.basename(path) for path in image_paths]
            source_paths = [os.path.basename(path) for path in source_paths]
            assert image_paths == source_paths
            image_batch = image_batch.to(self.env_args.device) * 2 - 1
            source_batch = source_batch.to(self.env_args.device) * 2 - 1
            qualities.append(metric(image_batch, source_batch))
        quality = torch.cat(qualities).mean()
        return quality.item()
