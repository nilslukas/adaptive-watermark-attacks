import torch
import torchvision.transforms as TF
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance
from tqdm import tqdm

from src.metrics.base_metric import BaseMetric


class FIDMetric(BaseMetric):
    """
    Needs 'pip install torchmetrics[image]'
    """

    def measure(self) -> float:
        """
        Measure the FID score.
        """
        assert self.quality_args.image_folder is not None, "FID needs to specify a folder with generated images"
        assert self.quality_args.reference_image_folder is not None, "FID needs to specify a folder with reference images"

        fid = FrechetInceptionDistance(self.quality_args.fid_features).to(self.env_args.device)
        fid.set_dtype(torch.float64)  # as recommended by the docs

        def calculate_fid(loader, is_real):
            """
            Calculate FID for a given set of images.

            Args:
                loader (DataLoader): DataLoader for the images.
                is_real (bool): True for real images, False for fake images.

            Returns:
                float: FID score.
            """
            for imgs, _ in tqdm(loader, desc=f"Calculating FID (real={is_real})"):
                imgs_uint8 = (imgs * 255).to(torch.uint8)
                fid.update(imgs_uint8, real=is_real)

        transform = TF.Compose([
            TF.Resize((299, 299)),
            TF.ToTensor()
        ])

        real_loader = self.get_image_loader(self.quality_args.reference_image_folder, transform)
        fake_loader = self.get_image_loader(self.quality_args.image_folder, transform)

        calculate_fid(real_loader, is_real=True)
        calculate_fid(fake_loader, is_real=False)

        print(f"> Computing FID .. ")
        quality = fid.compute()
        print(
            f"> Measured FID: {quality} for {len(fake_loader.dataset)} fake and {len(real_loader.dataset)} real images")
        return quality.item()
