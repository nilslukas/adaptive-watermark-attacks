import os.path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.arguments.adaptive_diffuser_args import AdaptiveDiffuserArgs
from src.arguments.evasion_args import AdaptiveDiffusionEvasionArgs
from src.datasets.image_folder import ImageFolderDataset
from src.evasion.evasion_attack import BlackBoxEvasionAttack
from src.models.img2img.adaptive_diffuser import AdaptiveDiffuser
from src.models.model_factory import ModelFactory
from src.utils.highlited_print import bcolors
from src.utils.utils import plot_images
from src.utils.web import download_and_unzip


class AdaptiveDiffusionAttack(BlackBoxEvasionAttack):
    """
    Train an image refiner using the reward model.
    """

    def load_adaptive_diffuser(self, ckpt: str) -> AdaptiveDiffuser:
        """
        Load the adaptive diffusion model
        """
        ckpt = download_and_unzip(ckpt)
        adaptive_diffuser_args: AdaptiveDiffuserArgs = torch.load(ckpt, map_location='cpu')[AdaptiveDiffuser.ADAPTIVE_DIFF_KEY]
        img2img = ModelFactory.from_adaptive_diffuser_args(adaptive_diffuser_args, env_args=self.env_args)
        img2img.load(torch.load(ckpt, map_location='cpu'))
        print(f"> Successfully loaded the adaptive diffuser from '{bcolors.OKGREEN}{os.path.abspath(ckpt)}{bcolors.ENDC}''.")
        return img2img.eval()

    @torch.no_grad()
    def attack(self, *args, **kwargs):
        """
        Train an adaptive diffuser against this watermark.
        """
        self.attack_args: AdaptiveDiffusionEvasionArgs
        assert self.attack_args.surr_diffusion_ckpt is not None, "Please specify a surrogate diffusion model."

        src_dataset = ImageFolderDataset(self.attack_args.root_folder)
        data_loader = DataLoader(src_dataset, shuffle=False, batch_size=self.env_args.batch_size)

        #  Setup the diffuser
        vae = self.load_adaptive_diffuser(self.attack_args.surr_diffusion_ckpt)
        vae.to(self.env_args.device)

        ctr = 0
        file_names = []
        before_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        diffused_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        for idx, (data, paths) in enumerate(tqdm(data_loader, disable=not self.attack_args.verbose_attack)):
            file_names += [os.path.basename(path) for path in paths]
            before_images[ctr:ctr + len(data)] = data
            data = data.to(self.env_args.device)

            encoded = data
            for _ in range(self.attack_args.diffusion_magnitude):
                encoded = torch.clamp(vae(encoded).cpu(), 0, 1).to(self.env_args.device)
            diffused_images[ctr:ctr + len(data)] = encoded
            ctr += len(data)

        # logging
        if self.attack_args.verbose_attack:
            plot_images(torch.cat([before_images[-4:], diffused_images[-4:]], 0), n_row=4,
                        title="AdvNoise Before/After")
        self.save_images(diffused_images, names=file_names)
        return diffused_images
