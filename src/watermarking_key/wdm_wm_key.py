import os

import torch
import wandb
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from src.arguments.wm_key_args import WDMWatermarkingKeyArgs, WatermarkingKeyArgs
from src.criteria.lpips_loss import LPIPSLoss
from src.models.autoencoders.stega import StegaStampDecoder, StegaStampEncoder
from src.models.generators.image_generator import ImageGenerator
from src.utils.highlited_print import bcolors
from src.utils.smoothed_value import SmoothedValue
from src.utils.utils import compute_bitwise_acc
from src.utils.web import download_and_unzip
from src.watermarking_key.wm_key import TrainableWatermarkingKey, MultiBitWatermarkingKey


class WDMWatermarkingKey(MultiBitWatermarkingKey, TrainableWatermarkingKey):
    """
    The WDM watermarking key. Can be used to generate watermarked images and to extract the watermark.
    """
    ENCODER_STATE_DICT_KEY = "wdm_encoder"  # the state dict of the encoder
    DECODER_STATE_DICT_KEY = "wdm_decoder"  # the state dict of the decoder

    def __init__(self, wm_key_args: WDMWatermarkingKeyArgs, **kwargs):
        super().__init__(wm_key_args=wm_key_args, **kwargs)
        self.wm_key_args: WDMWatermarkingKeyArgs
        self.decoder = None  # the inversion pipe (maps [image] -> [message])
        self.encoder = None  # the embedding pipe (maps [image, message] -> [watermarked image])

    def embed(self, generator: ImageGenerator, message: torch.Tensor) -> ImageGenerator:
        """
        Embeds one message into a generator's parameters and returns an image generator that only
        produces watermarked images.
        """
        raise NotImplementedError

    def keygen(self, generator: ImageGenerator = None):
        """
        Generate a watermarking key. Optionally accepts a generator
        """
        self.wm_key_args: WDMWatermarkingKeyArgs

        assert generator is not None, "WDM requires the specification of a generator to generate a key."
        self.print_shield()  # inform user that training will begin.

        # Setup the logger
        run = self.get_logger()

        self.decoder = StegaStampDecoder(self.wm_key_args.bitlen, model_type=self.wm_key_args.decoder_arch)
        self.decoder.to(self.env_args.device)
        self.encoder = StegaStampEncoder(resolution=512,  # ToDo: Make this variable
                                         fingerprint_size=self.wm_key_args.bitlen)  # we upscale to 256 because it needs to be a power of 2
        # decoder will resize to 224 anyways
        self.encoder.to(self.env_args.device)

        # get the decoder and define optimizer
        opt = Adam(list(self.encoder.parameters()) +
                   list(self.decoder.parameters()), lr=self.wm_key_args.lr)

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
        generator.pipe, encoder, decoder, self.data_loader, opt = accelerator.prepare(
            generator.pipe,
            self.encoder,
            self.decoder,
            data_loader,
            opt)

        bce = BCEWithLogitsLoss()
        lpips = LPIPSLoss() if self.wm_key_args.starter_keygen_lambda_lpips > 0 else None

        bit_acc = SmoothedValue()

        starter_keygen_lambda_lpips = self.wm_key_args.starter_keygen_lambda_lpips

        heatup_period = self.wm_key_args.heatup_period  # number of steps to progressively increase the lambda lpips over

        count, heatup_triggered = 0, False
        with tqdm(total=self.env_args.log_every,
                  desc="Training WDM Key (infinity loop)") as pbar:
            step = 0
            while True:
                loss_dict = {"iter": step}

                generator.eval()
                encoder.train()
                decoder.train()

                with torch.no_grad():
                    seed, w, x = next(self.load_next_batch())  # infinity loader
                    msg = self.sample_message(n=w.shape[0]).to(self.env_args.device).float()

                x_wm = torch.clamp(encoder(x, message=msg), 0, 1)
                extracted_msg = decoder(preprocessing(x_wm))
                bit_acc.update(compute_bitwise_acc(msg, torch.sigmoid(extracted_msg)))

                loss_bce = bce(extracted_msg, msg)
                loss_dict['loss_bce'] = float(loss_bce)
                loss = loss_bce

                if self.wm_key_args.l1_loss_coef > 0:
                    loss_l1 = self.wm_key_args.l1_loss_coef * (x - x_wm).abs().mean()
                    loss_dict['loss_l1'] = float(loss_l1)
                    loss += loss_l1

                if heatup_triggered and count >= self.wm_key_args.heatup_period:
                    # switch on heatup once ae is capable
                    loss_l_inf = torch.pow(torch.abs(x - x_wm) / float(self.wm_key_args.l_inf_epsilon),
                                           6).mean() * self.wm_key_args.l_inf_loss_coef
                    loss_dict['loss_l_inf'] = float(loss_l_inf)
                    loss += loss_l_inf

                if self.wm_key_args.keygen_lambda_lpips > 0:
                    if heatup_triggered and count < heatup_period:
                        delta_coef = self.wm_key_args.keygen_lambda_lpips - starter_keygen_lambda_lpips
                        loss_lpips = (starter_keygen_lambda_lpips + (delta_coef * float(count / heatup_period))) * \
                                     lpips(x, x_wm).mean()
                        count += 1
                    elif heatup_triggered and count >= heatup_period:
                        loss_lpips = self.wm_key_args.keygen_lambda_lpips * lpips(x, x_wm).mean()

                    else:
                        loss_lpips = starter_keygen_lambda_lpips * lpips(x, x_wm).mean()

                    loss_dict['loss_lpips'] = float(loss_lpips)
                    loss += loss_lpips

                accelerator.backward(loss)

                if step == 0:  # Check if gradients are available
                    print()
                    self.gradient_check(decoder, name="decoder")
                    self.gradient_check(encoder, name="encoder")

                if (step + 1) % self.env_args.gradient_accumulation_steps == 0:
                    opt.step()
                    opt.zero_grad()

                bit_acc.update(compute_bitwise_acc(msg, torch.sigmoid(extracted_msg)))
                loss_dict['bit_acc'] = bit_acc.avg
                loss_dict['capacity'] = max(0, 2 * (
                        (bit_acc.avg / 100) * self.wm_key_args.bitlen - 0.5 * self.wm_key_args.bitlen))
                loss_dict['total_loss'] = loss.item()
                if self.env_args.logging_tool == "wandb":
                    wandb.log({**loss_dict})  # log to wandb

                if loss_dict['capacity'] > self.wm_key_args.capacity_trigger:
                    heatup_triggered = True

                # Logging
                if step % self.env_args.log_every == 0:
                    with torch.no_grad():
                        self.log_data(step=step, bit_acc=bit_acc, extracted_msg=extracted_msg,
                                      msg=msg, pbar=pbar, x=x, x_wm=x_wm, loss_dict=loss_dict, run=run)

                # Saving
                if self.wm_key_args.key_ckpt is not None and step > 0 and step % self.env_args.save_every == 0:
                    self.save(self.wm_key_args.key_ckpt)

                step += 1
                pbar.update(1)
                pbar.set_description(
                    f"Training WDM Key (Step {step}, ∞ loop), Acc: {bit_acc.avg:.2f}, loss: {loss.item():.3f}")

    def load(self, ckpt=None):
        """
        Load from a checkpoint
        """
        ckpt = ckpt if ckpt is not None else torch.load(download_and_unzip(self.wm_key_args.key_ckpt),
                                                        map_location='cpu')
        if isinstance(ckpt, str):
            ckpt = torch.load(ckpt, map_location='cpu')
        self.wm_key_args: WDMWatermarkingKeyArgs
        ckpt_before = download_and_unzip(self.wm_key_args.key_ckpt) if self.wm_key_args.key_ckpt is not None else None
        self.wm_key_args = ckpt[WatermarkingKeyArgs.WM_KEY_ARGS_KEY]
        self.wm_key_args.key_ckpt = ckpt_before

        self.decoder = StegaStampDecoder(self.wm_key_args.bitlen, model_type=self.wm_key_args.decoder_arch)
        self.decoder.load_state_dict(ckpt[self.DECODER_STATE_DICT_KEY])

        self.encoder = StegaStampEncoder(resolution=512,  # ToDo: Make this variable
                                         fingerprint_size=self.wm_key_args.bitlen)
        self.encoder.load_state_dict(ckpt[self.ENCODER_STATE_DICT_KEY])
        print(
            f"> Successfully loaded {bcolors.OKGREEN}WDM{bcolors.ENDC} from '{bcolors.OKGREEN}{self.wm_key_args.key_ckpt}{bcolors.ENDC}'.")
        return self

    @torch.no_grad()
    def extract(self, x: torch.Tensor):
        """
        Extracts the message from a watermarked image.
        """
        self.decoder.eval().to(x.device)
        return torch.round(torch.sigmoid(self.decoder(x)))

    def extract_message_with_gradients(self, x):
        self.decoder.eval().to(x.device)
        return self.decoder(x)

    def verify_message_with_gradients(self, y_pred, y_true):
        if y_true.shape[0] != len(y_pred):
            y_true = y_true.repeat_interleave(len(y_pred), 0)
        return -BCEWithLogitsLoss()(y_pred, 1-y_true)

    def save(self, ckpt_fn: str = None) -> dict:
        """
        Saves the key into a checkpoint file '*.pt'
        """
        self.wm_key_args: WDMWatermarkingKeyArgs
        accelerator = self.env_args.get_accelerator()
        save_dict = {
            WatermarkingKeyArgs.WM_KEY_ARGS_KEY: self.wm_key_args,
            self.ENCODER_STATE_DICT_KEY: self.encoder.state_dict(),
            self.DECODER_STATE_DICT_KEY: self.decoder.state_dict()
        }
        if ckpt_fn is not None and accelerator.is_local_main_process:
            accelerator.save(save_dict, ckpt_fn)
            print(f"> Writing a WDM key to '{bcolors.OKGREEN}{os.path.abspath(ckpt_fn)}{bcolors.ENDC}'.")
        return save_dict
