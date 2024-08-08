import os

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from src.arguments.wm_key_args import PTWWatermarkingKeyArgs
from src.criteria.lpips_loss import LPIPSLoss
from src.models.generators.image_generator import ImageGenerator
from src.models.model_factory import ModelFactory
from src.utils.highlited_print import bcolors, print_warning
from src.utils.smoothed_value import SmoothedValue
from src.utils.web import download_and_unzip
from src.watermarking_key.wm_key import MultiBitWatermarkingKey, TrainableWatermarkingKey


class PTWWatermarkingKey(MultiBitWatermarkingKey, TrainableWatermarkingKey):
    """
    The PTW watermarking key
    """
    MODEL_STATE_DICT_KEY = "ptw_state_dict"  # the key in the dictionary to save the model state

    def __init__(self, wm_key_args: PTWWatermarkingKeyArgs, **kwargs):
        super().__init__(wm_key_args=wm_key_args, **kwargs)
        self.wm_key_args: PTWWatermarkingKeyArgs

        self.cache = {
            'mappers': None,
            'logging_data': {},
            'data_loader': None
        }

        self.decoder = ModelFactory.load_decoder(self.wm_key_args.bitlen, decoder_arch=self.wm_key_args.decoder_arch)
        if self.wm_key_args.key_ckpt is not None and os.path.exists(self.wm_key_args.key_ckpt):
            self.load(self.wm_key_args.key_ckpt)  # restore state if possible

    def embed(self, generator: ImageGenerator, message: torch.Tensor):
        """
        Embeds a message into a generator's parameters.
        """
        raise NotImplementedError

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts an embedded_message from images.
        """
        self.decoder.eval()
        msg = torch.round(torch.sigmoid(self.decoder(x)))
        return msg

    def keygen(self, generator=None):
        """
        Generate a watermarking key. Optionally accepts a generator
        """
        self.wm_key_args: PTWWatermarkingKeyArgs
        assert generator is not None, "PTW requires the specification of a generator to generate a key."
        self.print_shield()  # inform user that training will begin.

        # Setup the logger
        run = self.get_logger()

        # Get the generator and inject watermarking layers
        g_target, mappers = ModelConverter3(self.wm_key_args).convert(generator)
        assert mappers.size() > 0, "Need to specify at least one mapper to embed a watermark!"

        # get the decoder and define optimizer
        opt = Adam([
            {'params': self.decoder.parameters(), 'lr': self.wm_key_args.lr_decoder},
            {'params': mappers.parameters(), 'lr': self.wm_key_args.lr_mapper}
        ])

        # Prepare the dataset
        data_loader = self.get_data_loader(generator)

        # preprocessors
        preprocessing = transforms.Compose([])
        if self.wm_key_args.add_preprocessing:
            preprocessing = transforms.Compose([
                transforms.RandomErasing(),
                transforms.RandomAffine(degrees=(-5, 5), translate=(0.01, 0.01), scale=(0.95, 1.0))
            ])

        # prepare all modules.
        accelerator = self.env_args.get_accelerator()
        g_target.pipe, g_target.model, decoder, data_loader, mappers, opt = accelerator.prepare(
            g_target.pipe,
            g_target.model,
            self.decoder,
            data_loader,
            mappers,
            opt)

        bce = BCEWithLogitsLoss()
        lpips = LPIPSLoss() if self.wm_key_args.keygen_lambda_lpips > 0 else None

        print(f"> Mapper Params:  {sum(p.numel() for p in mappers.parameters()) / 10 ** 6:.2f} million")
        print(f"Mapper size: {mappers.size()}")

        bit_acc = SmoothedValue()
        with tqdm(total=self.env_args.log_every,
                  desc="Training PTW Key (infinity loop)") as pbar:
            step = 0
            while True:
                loss_dict = {"iter": step}

                g_target.model.eval()
                decoder.train()

                with torch.no_grad():
                    seed, w, x = next(self.load_next_batch())  # infinity loader
                    msg = self.sample_message(n=w.shape[0]).to(self.env_args.device).float()
                    mappers.set_msg(msg)  # forward the embedded_message to the mappers

                _, x_wm = g_target.generate(w=w, seed=seed,
                                            num_images=self.env_args.batch_size,
                                            mapper=mappers,
                                            use_gradients_after=self.wm_key_args.keygen_steps)

                loss_dict['mapper_magnitude']: float = sum(
                    [abs(mapper.get_magnitude()) for mapper in mappers]) / mappers.size()

                extracted_msg = decoder(preprocessing(x_wm))  # extract the message

                loss_bce = bce(extracted_msg, msg)
                loss_dict['loss_bce'] = float(loss_bce)
                loss = loss_bce

                if self.wm_key_args.keygen_lambda_lpips > 0:
                    loss_lpips = self.wm_key_args.keygen_lambda_lpips * lpips(x, x_wm).mean()
                    loss_dict['loss_lpips'] = float(loss_lpips)
                    loss += loss_lpips

                accelerator.backward(loss)

                if step == 0 or step == 10:  # Check if gradients are available
                    self.gradient_check(mappers, decoder)

                if (step + 1) % self.env_args.gradient_accumulation_steps == 0:
                    opt.step()
                    opt.zero_grad()

                # Logging
                if step % self.env_args.log_every == 0:
                    with torch.no_grad():
                        self.log_data(step=step, bit_acc=bit_acc, extracted_msg=extracted_msg,
                                      msg=msg, pbar=pbar, x=x, x_wm=x_wm, loss_dict=loss_dict, run=run)

                # Saving
                if self.wm_key_args.keygen_steps is not None and step > 0 and step % self.env_args.save_every == 0:
                    if os.path.exists(self.wm_key_args.keygen_steps):
                        print_warning(f"> Checkpoint path already exists. Will overwrite the checkpoint.")
                    ckpt_fn = self.wm_key_args.key_ckpt
                    self.save(ckpt_fn)

                step += 1
                pbar.update(1)
                pbar.set_description(
                    f"Training PTW Key (Step {step}, ∞ loop), Acc: {bit_acc.avg:.2f}, loss: {loss_dict['loss_bce']:.3f}")

    def load(self, ckpt_fn: str = None) -> None:
        """ Load the state from a checkpoint. """
        ckpt_fn = self.wm_key_args.key_ckpt if ckpt_fn is None else ckpt_fn
        ckpt_fn = download_and_unzip(ckpt_fn)
        data = torch.load(ckpt_fn)
        self.load_state_dict(
            {key.replace('module.', ''): value for key, value in data[self.MODEL_STATE_DICT_KEY].items()}, strict=False)
        self.wm_key_args.key_ckpt = ckpt_fn
        print(
            f"> Restored the state from {bcolors.OKGREEN}{os.path.abspath(self.wm_key_args.key_ckpt)}{bcolors.ENDC}. ")

    def save(self, ckpt_fn: str = None) -> dict:
        """ Create a single '*.pt' file. Returns only the dict if no ckpt_fn is provided.  """
        accelerator = self.env_args.get_accelerator()
        save_dict = {
            self.MODEL_STATE_DICT_KEY: self.state_dict(),
            **super().save()
        }
        if ckpt_fn is not None and accelerator.is_local_main_process:
            print(f"> Writing a PTW decoder to '{bcolors.OKGREEN}{os.path.abspath(ckpt_fn)}{bcolors.ENDC}'.")
            accelerator.save(save_dict, ckpt_fn)
        return save_dict
