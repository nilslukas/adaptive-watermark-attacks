import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.arguments.evasion_args import AdvNoiseEvasionArgs
from src.datasets.image_folder import ImageFolderDataset
from src.evasion.evasion_attack import BlackBoxEvasionAttack
from src.utils.utils import plot_images, compute_bitwise_acc
from src.utils.web import download_and_unzip
from src.watermarking_key.wm_key import WatermarkingKey
from src.watermarking_key.wm_key_factory import WatermarkingKeyFactory


def generate_adversarial_noise(x_wm: torch.Tensor,
                               wm_key: WatermarkingKey,
                               steps: int = 5,
                               lr: float = 0.01,
                               epsilon: float = 4 / 255):
    """
    Generate adversarial noise based on true messages and the surrogate watermarking model.
    """

    x_wm = x_wm.to(wm_key.env_args.device)
    noise = torch.zeros_like(x_wm, device=wm_key.env_args.device)
    noise.requires_grad = True
    y_true = None

    opt = optim.Adam([noise], lr=lr)

    for _ in range(steps):
        opt.zero_grad()

        if y_true is None:
            # On the first iteration, determine the true watermark (message) based on the current prediction
            with torch.no_grad():
                y_true = wm_key.extract(x_wm)

        # Forward pass through the watermarking key (surrogate model)
        y_pred = wm_key.extract_message_with_gradients(torch.clamp(x_wm + noise, 0, 1))

        # Compute the loss between the inverted true message and the current prediction
        loss = -wm_key.verify_message_with_gradients(y_pred, y_true)

        # Backward pass and optimization
        loss.backward()
        opt.step()

        # Clip the noise to ensure it is within a valid range (e.g., [-epsilon, epsilon])
        noise.data = torch.clamp(noise.data, -epsilon, epsilon)

    return torch.clamp(x_wm + noise, 0, 1).detach()


class AdvNoiseAttack(BlackBoxEvasionAttack):
    """
    Generate adversarial noise against the surrogate key.
    """

    def attack(self, *args, **kwargs):
        """
        Compute the adversarial noise and apply it to images to generate adversarial examples.
        """
        self.attack_args: AdvNoiseEvasionArgs
        assert self.attack_args.surr_key_ckpt is not None, "Adaptive attacks need to specify a surrogate key."

        src_dataset = ImageFolderDataset(self.attack_args.root_folder)
        data_loader = DataLoader(src_dataset, shuffle=False, batch_size=self.env_args.batch_size)

        print(f"> Found {len(src_dataset)} entries, and {len(data_loader)}")
        wm_key: WatermarkingKey = WatermarkingKeyFactory.from_checkpoint(self.attack_args.surr_key_ckpt,
                                                                         env_args=self.env_args)
        wm_key.load(torch.load(download_and_unzip(self.attack_args.surr_key_ckpt), map_location='cpu'))
        try:  # ToDo_ Find better solution
            wm_key.wm_key_args.reversal_inference_steps = 2
        except:
            pass
        print(f"> (AdvNoise) Loaded surrogate key from {self.attack_args.surr_key_ckpt} with eps={self.attack_args.adaptive_noise_epsilon}")

        ctr = 0
        file_names = []
        before_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        noised_images = torch.empty((len(src_dataset), *src_dataset[0][0].shape))
        for data, paths in tqdm(data_loader, disable=not self.attack_args.verbose_attack):
            file_names += [os.path.basename(path) for path in paths]
            before_images[ctr:ctr + len(data)] = data
            noised_images[ctr:ctr + len(data)] = generate_adversarial_noise(data, wm_key,
                                                                            steps=self.attack_args.adaptive_noise_opt_steps,
                                                                            lr=self.attack_args.adaptive_noise_lr,
                                                                            epsilon=self.attack_args.adaptive_noise_epsilon)
            #plot_images(noised_images[ctr:ctr + len(data)])

            ctr += len(data)
        if self.attack_args.verbose_attack:
            plot_images(torch.cat([before_images[-4:], noised_images[-4:]], 0), n_row=4,
                        title="AdvNoise Before/After")
        self.save_images(noised_images, names=file_names)
        return noised_images
