import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from imwatermark import WatermarkEncoder, WatermarkDecoder
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms
from tqdm import tqdm

from src.arguments.wm_key_args import WDMWatermarkingKeyArgs, WatermarkingKeyArgs, SDWWatermarkingKeyArgs
from src.datasets.image_folder import ImageFolderDataset
from src.models.autoencoders.stega import StegaStampDecoder
from src.models.generators.image_generator import ImageGenerator
from src.utils.highlited_print import bcolors
from src.utils.smoothed_value import SmoothedValue
from src.utils.utils import compute_bitwise_acc, plot_images
from src.utils.web import download_and_unzip
from src.watermarking_key.wm_key import MultiBitWatermarkingKey


class SDWWatermarkingKey(MultiBitWatermarkingKey):
    """
    The Stable-Diffusion-Watermark used by the stabilityAI people.
    Repo: https://github.com/ShieldMnt/invisible-watermark
    """
    DECODER_STATE_DICT_KEY = "sdw_decoder"  # the state dict of the decoder

    def __init__(self, wm_key_args: SDWWatermarkingKeyArgs, **kwargs):
        super().__init__(wm_key_args=wm_key_args, **kwargs)
        self.wm_key_args: SDWWatermarkingKeyArgs
        self.encoder = WatermarkEncoder()  # the encoding pipe (maps [message, image] -> image]
        self.encoder.loadModel()  # always load the model
        self.decoder = None  # the inversion pipe (maps [image] -> [message])
        self.original_decoder = WatermarkDecoder('bits', self.wm_key_args.bitlen)

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract the watermark from an image.
        """
        y_pred = torch.empty((x.shape[0], self.wm_key_args.bitlen), dtype=torch.float32, device=self.env_args.device)

        for i, img in enumerate(x):
            img = (img.cpu().permute((1, 2, 0)).numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            msg = self.original_decoder.decode(img, self.wm_key_args.wm_method)  # list of bits
            y_pred[i] = torch.tensor(msg, dtype=torch.float32)
        return y_pred

    def keygen(self, generator: ImageGenerator = None):
        """
        Generate images with the watermark and generate a differentiable decoder.
        """
        assert generator is not None, "Needs to provide a generator to generate images."
        self.wm_key_args: SDWWatermarkingKeyArgs
        self.wm_key_args.message = self.sample_message(1)
        bce_criterion = BCEWithLogitsLoss()

        if self.wm_key_args.resume_from is not None:
            self.load(torch.load(download_and_unzip(self.wm_key_args.resume_from), map_location='cpu'))
        else:
            self.decoder = StegaStampDecoder(self.wm_key_args.bitlen, model_type=self.wm_key_args.decoder_arch)
            self.decoder.to(self.env_args.device)

        opt = torch.optim.Adam(self.decoder.parameters(), lr=self.wm_key_args.lr)
        self.save(self.wm_key_args.key_ckpt)

        @torch.no_grad()
        def next_batch():
            if self.wm_key_args.dataset_root is not None:
                print(f"> Loading dataset from {self.wm_key_args.dataset_root}")
                # dataset root
                dataset = ImageFolderDataset(self.wm_key_args.dataset_root)
                data_loader = self.env_args.make_data_loader(dataset)

                preprocessing = transforms.Compose([
                    transforms.RandomErasing(),
                    transforms.RandomAffine(degrees=(-5, 5), translate=(0.01, 0.01), scale=(0.95, 1.0))
                ])

                while True:
                    for images, _ in data_loader:
                        y_true = self.sample_message(self.env_args.batch_size)
                        out = []

                        images = images.cpu().numpy().transpose(0, 2, 3, 1) * 255
                        for img, msg in zip(images, y_true):
                            self.encoder.set_watermark('bits', msg)
                            img = Image.fromarray(img.astype(np.uint8))
                            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            img = self.encoder.encode(img, self.wm_key_args.wm_method)
                            img = Image.fromarray(img[:, :, ::-1])
                            img = transforms.ToTensor()(img)
                            out += [img]
                        images = torch.stack(out, dim=0).to(self.env_args.device)
                        yield preprocessing(images), y_true
            # generate on the fly
            while True:
                print(f"> Generating watermarked images on the fly ")
                y_true = self.sample_message(self.env_args.batch_size)
                self.set_message(y_true)  # generate a random msg
                _, x_wm = generator.generate(num_images=self.env_args.batch_size)  # gen rnd img
                yield x_wm, y_true

        generator.set_watermarking_key(self)
        bit_acc = SmoothedValue()
        step = 0
        with tqdm(total=self.env_args.save_every) as pbar:
            for x_wm, y_true in next_batch():
                self.decoder.train()
                y_pred = self.decoder(x_wm)
                #y_true = self.extract(x_wm)  # read back what the decoder thinks is in the image
                loss = bce_criterion(y_pred, y_true)
                loss.backward()
                opt.step()
                opt.zero_grad()

                bit_acc.update(compute_bitwise_acc(y_true, y_pred))
                pbar.set_description(f" Step: {step} Loss: {loss.item():.4f}, BitAcc: {bit_acc:.4f} (save at {self.env_args.save_every})")
                pbar.update(1)

                if step % self.env_args.save_every == 0:
                    self.save(self.wm_key_args.key_ckpt)

                if step % 100 == 0:
                    plot_images(x_wm, title=self.wm_key_args.wm_method)
                step += 1
                pbar.update(1)
                if step % self.env_args.save_every == 0:
                    pbar.reset()

    def extract_message_with_gradients(self, x):
        self.decoder.eval().to(x.device)
        return self.decoder(x)

    def verify_message_with_gradients(self, y_pred, y_true):
        if y_true.shape[0] != len(y_pred):
            y_true = y_true.repeat_interleave(len(y_pred), 0)
        loss = -BCEWithLogitsLoss()(y_pred, 1-y_true)
        return loss

    def save(self, ckpt_fn: str = None) -> dict:
        """
        Saves the key into a checkpoint file '*.pt'
        """
        print(f"> Entering save!")
        self.wm_key_args: SDWWatermarkingKeyArgs
        accelerator = self.env_args.get_accelerator()
        save_dict = {
            WatermarkingKeyArgs.WM_KEY_ARGS_KEY: self.wm_key_args,
            self.DECODER_STATE_DICT_KEY: self.decoder.state_dict()
        }
        if ckpt_fn is not None and accelerator.is_local_main_process:
            accelerator.save(save_dict, ckpt_fn)
            print()
            print(
                f"> Writing a {bcolors.OKBLUE}SDW - {self.wm_key_args.wm_method}{bcolors.ENDC} key to '{bcolors.OKGREEN}{os.path.abspath(ckpt_fn)}{bcolors.ENDC}'.")
        return save_dict

    def load(self, ckpt=None):
        """
        Load from a checkpoint
        """
        ckpt = ckpt if ckpt is not None else torch.load(download_and_unzip(self.wm_key_args.key_ckpt), map_location='cpu')
        self.wm_key_args: WDMWatermarkingKeyArgs
        self.wm_key_args = ckpt[WatermarkingKeyArgs.WM_KEY_ARGS_KEY]
        self.original_decoder = WatermarkDecoder('bits', self.wm_key_args.bitlen)

        self.decoder = StegaStampDecoder(self.wm_key_args.bitlen, model_type=self.wm_key_args.decoder_arch)
        self.decoder.load_state_dict(ckpt[self.DECODER_STATE_DICT_KEY])

        self.decoder.to(self.env_args.device)
        print(
            f"> Successfully loaded {bcolors.OKGREEN}SDM{bcolors.ENDC}.")
        return self
