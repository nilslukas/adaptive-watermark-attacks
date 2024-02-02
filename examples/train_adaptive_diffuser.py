import torch
import transformers
from torch.optim import SGD, Adam
from tqdm import tqdm

from src.arguments.adaptive_diffuser_args import AdaptiveDiffuserArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.criteria.lpips_loss import LPIPSLoss
from src.models.model_factory import ModelFactory
from src.utils.highlited_print import bcolors
from src.utils.smoothed_value import SmoothedValue
from src.utils.utils import plot_images
from src.utils.web import download_and_unzip
from src.watermarking_key.wm_key import WatermarkingKey
from src.watermarking_key.wm_key_factory import WatermarkingKeyFactory


def parse_args():
    """ Trains a surrogate watermark decoder on a set of labeled watermarking keys.
    """
    parser = transformers.HfArgumentParser((AdaptiveDiffuserArgs,
                                            WatermarkingKeyArgs,
                                            ModelArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def train_adaptive_reward_model(
        adaptive_diffuser_args: AdaptiveDiffuserArgs,
        wm_key_args: WatermarkingKeyArgs,  # The surrogate watermarking key
        model_args: ModelArgs,  # The surrogate diffusion model
        env_args: EnvArgs,
        config_args: ConfigArgs):
    """
    Given a surrogate watermarking key, train an image-to-image transformation model that adversarially flips the reward.
    """
    if config_args.exists():
        adaptive_diffuser_args = config_args.get_adaptive_diffuser_args()
        wm_key_args = config_args.get_watermarking_key_args()
        model_args = config_args.get_model_args()
        env_args = config_args.get_env_args()

    # load a generator capable of generating watermarked images containing any message.
    surrogate_generator = ModelFactory.from_model_args(model_args, wm_key_args=wm_key_args,
                                                       env_args=env_args).load()
    surrogate_generator.pipe.requires_safety_checker = False

    wm_key: WatermarkingKey = WatermarkingKeyFactory.from_watermark_key_args(wm_key_args, env_args=env_args).load()

    try: # ToDo_ Find better solution
        wm_key.wm_key_args.reversal_inference_steps = adaptive_diffuser_args.num_reversal_inference_steps
    except:
        pass

    with torch.no_grad():
        _, x_nowm = surrogate_generator.generate(num_images=4)
        acc = wm_key.verify(wm_key.extract(x_nowm), wm_key.sample_message(1))['accuracy']
        print(f"> Non-Watermarked Detection Accuracy = {bcolors.OKGREEN}{100*acc:.2f}%{bcolors.ENDC}")

    # watermark the surrogate generator.
    surrogate_generator.set_watermarking_key(wm_key)

    img2img = ModelFactory.from_adaptive_diffuser_args(adaptive_diffuser_args, env_args=env_args)
    if adaptive_diffuser_args.resume_from is not None:
        img2img.load(torch.load(download_and_unzip(adaptive_diffuser_args.resume_from), map_location='cpu'))

    # train the vae to remove the watermark.
    surrogate_generator.eval()
    img2img.img2img.train()
    if adaptive_diffuser_args.opt.lower() == "sgd":
        opt = SGD(img2img.img2img.parameters(), lr=adaptive_diffuser_args.lr)
    elif adaptive_diffuser_args.opt.lower() == "adam":
        opt = Adam(img2img.img2img.parameters(), lr=adaptive_diffuser_args.lr) # adam gives infinite gradients
    else:
        raise ValueError(f"Unknown optimizer: {adaptive_diffuser_args.optimizer}")
    img2img.train().to(env_args.device)

    lpips_loss = LPIPSLoss()  # preserve perceptual quality

    accelerator = env_args.get_accelerator()
    wm_key, img2img, surrogate_generator, opt = accelerator.prepare(wm_key, img2img, surrogate_generator, opt)

    lambda_lpips = adaptive_diffuser_args.lambda_lpips
    lambda_bce = adaptive_diffuser_args.lambda_bce

    wm_key: WatermarkingKey
    step = 0
    acc_before, acc_after = SmoothedValue(), SmoothedValue()
    with tqdm() as pbar:
        while True:
            with torch.no_grad():
                y_true = wm_key.sample_message(env_args.batch_size)
                wm_key.set_message(y_true)  # set a random message
                w, x_wm = surrogate_generator.generate(num_images=env_args.batch_size)  # generate b wm imgs
            x_e = img2img(x_wm).float()

            loss = torch.Tensor([0]).to(env_args.device)

            l_lpips = lpips_loss(x_e, x_wm.detach()).mean()
            loss += lambda_lpips * l_lpips
            l_percept = 1 * adaptive_diffuser_args.lambda_mse * torch.nn.functional.mse_loss(x_e, x_wm.detach())
            loss += l_percept

            y_pred = wm_key.extract_message_with_gradients(torch.clamp(x_e, 0, 1))
            l_bce = -lambda_bce * wm_key.verify_message_with_gradients(y_pred, y_true)
            loss += l_bce

            accelerator.backward(loss)

            if step == 0:
                for param in img2img.img2img.parameters():
                    print(param.mean())

            opt.step()
            opt.zero_grad()


            with torch.no_grad():
                before_acc = wm_key.verify(wm_key.extract(torch.clamp(x_wm, 0, 1)), y_true)['accuracy']
                acc_before.update(before_acc)
                acc_after.update(wm_key.verify(wm_key.extract(torch.clamp(x_e, 0, 1)), y_true)['accuracy'])

            if step % 10 == 0:
                plot_images(torch.clamp(torch.cat([x_wm.detach().cpu().float()[:3], x_e.detach().cpu().float()[:3], (x_e-x_wm).float().detach().cpu()[:3]]), 0, 1),
                            n_row=min(env_args.batch_size, 3), title=f"Detection Accuracy After {acc_after.avg*100:.2f}%")

            if step % env_args.save_every == 0 and step > 0:
                print(f"> (Acc_After={acc_after.avg}) Saving model to {bcolors.OKGREEN}{adaptive_diffuser_args.save_path}{bcolors.ENDC}")
                img2img.save(adaptive_diffuser_args.save_path)

            pbar.set_description(f"> Step: {step} Acc Before Attack: {acc_before.avg*100:.2f}%, "
                                 f"Acc After Attack: {acc_after.avg*100:.2f}% LPIPS: {l_lpips:.4f}, "
                                 f"Loss Percept: {l_percept:.4f}, Loss BCE: {l_bce:.4f}")
            step += 1
            pbar.update(1)



if __name__ == "__main__":
    train_adaptive_reward_model(*parse_args())
