"""CLI for inverting images into StyleGANv2's latent space"""
import os
import pickle
from pathlib import Path

import click
import torch
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from loho import get_lr, latent_noise
from losses.noise_loss import noise_normalize_, noise_regularize
from networks import lpips
from networks.style_gan_2 import Generator
from utils import image_utils


@click.group()
def cli():
    """CLI"""


def _setup_styleganv2(gan_output_size: int, styleganv2_ckpt_path: str):
    styleganv2_gen = Generator(gan_output_size, 512, 8)
    styleganv2_gen.load_state_dict(
        torch.load(styleganv2_ckpt_path)["g_ema"], strict=False
    )
    styleganv2_gen.eval()
    styleganv2_gen.cuda()

    return styleganv2_gen


@cli.command()
@click.option(
    "--gan-output-size",
    default=1024,
    type=int,
    help="Resolution of StyleGANv2 synthesized image output",
)
@click.option("--image-path", required=True, type=str, help="Path to image to invert")
@click.option("--initial-learning-rate", default=0.1, type=float, help="Learning rate")
@click.option(
    "--lpips-input-size",
    default=256,
    type=int,
    help="Resolution of image input to LPIPS loss",
)
@click.option(
    "--n-mean-latent",
    default=10000,
    type=int,
    help="Number of random samples to compute mean latent",
)
@click.option(
    "--n-steps",
    default=2000,
    type=int,
    help="Number of optimization steps",
)
@click.option("--noise-ramp", default=0.75, type=float, help="Noise ramp")
@click.option(
    "--noise-reg-loss-coeff",
    default=1e5,
    type=float,
    help="Noise regularization loss coefficient",
)
@click.option(
    "--noise-strength-coeff",
    default=0.05,
    type=float,
    help="Noise strength coefficient",
)
@click.option(
    "--output-dir",
    default=os.path.join("data", "results"),
    type=str,
    help="Directory to output inverted latent and reconstructed image to",
)
@click.option(
    "--save-pickle/--no-save-pickle",
    default=False,
    type=bool,
    help="Save pickles of latents, noises, and losses?",
)
@click.option(
    "--save-synth-every",
    default=500,
    type=int,
    help="Save synthesized image every N steps",
)
@click.option(
    "--styleganv2-ckpt-path",
    default=os.path.join("checkpoints", "stylegan2-ffhq-config-f.pt"),
    type=str,
    help="Path to pretrained StyleGANv2 weights",
)
@click.option(
    "--optimize-extended-w-space/--no-optimize-extended-w-space",
    default=True,
    type=bool,
    help="Optimize W+ space? If False optimizes W space",
)
def gan_invert(
    gan_output_size: int,
    image_path: str,
    initial_learning_rate: float,
    lpips_input_size: int,
    n_mean_latent: int,
    n_steps: int,
    noise_ramp: float,
    noise_reg_loss_coeff: float,
    noise_strength_coeff: float,
    output_dir: str,
    save_pickle: bool,
    save_synth_every: int,
    styleganv2_ckpt_path: str,
    optimize_extended_w_space: bool,
):
    """Invert image to latent"""
    os.makedirs(output_dir, exist_ok=True)

    input_img = Image.open(image_path).convert("RGB")
    input_img = transforms.Resize((lpips_input_size, lpips_input_size))(input_img)

    input_img = transforms.ToTensor()(input_img)
    input_img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(input_img)
    input_img = input_img.cuda()
    input_img = input_img.unsqueeze(0)

    image_utils.writeImageToDisk([input_img], ["input_img.png"], output_dir)

    lpips_loss_func = lpips.PerceptualLoss(
        model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], use_gpu=True
    )

    styleganv2_gen = _setup_styleganv2(gan_output_size, styleganv2_ckpt_path)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device="cuda")
        latent_out = styleganv2_gen.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    noises_single = styleganv2_gen.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(input_img.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(input_img.shape[0], 1)
    latent_in = latent_in.unsqueeze(1)
    if optimize_extended_w_space:
        latent_in = latent_in.repeat(1, styleganv2_gen.n_latent, 1)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = torch.optim.Adam([latent_in] + noises, lr=initial_learning_rate)

    losses_log = {"lpips": [], "noise": []}
    progress_bar = tqdm(range(n_steps))
    for step_i in progress_bar:
        step_fraction = step_i / n_steps
        optimizer.param_groups[0]["lr"] = get_lr(step_fraction, initial_learning_rate)

        if not optimize_extended_w_space:
            latent_repeated = latent_in.repeat(1, styleganv2_gen.n_latent, 1)
        else:
            latent_repeated = latent_in

        noise_strength = (
            latent_std
            * noise_strength_coeff
            * max(0, 1 - (step_fraction / noise_ramp)) ** 2
        )
        latent_n = latent_noise(latent_repeated, noise_strength.item())

        synth_img, _ = styleganv2_gen([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = synth_img.shape

        if height > lpips_input_size:
            factor = height // lpips_input_size

            synth_img = synth_img.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            synth_img = synth_img.mean(dim=[3, 5])

        lpips_loss_value = lpips_loss_func(synth_img, input_img)
        losses_log["lpips"].append(lpips_loss_value.item())

        noise_loss = noise_reg_loss_coeff * noise_regularize(noises)
        losses_log["noise"].append(noise_loss.item())

        optimizer.zero_grad()
        (lpips_loss_value + noise_loss).backward()
        optimizer.step()

        noise_normalize_(noises)

        progress_bar.set_description(
            f"lpips: {lpips_loss_value.item():.4f}; noise: {noise_loss.item():.4f}"
        )

        if ((step_i + 1) % save_synth_every) == 0:
            image_utils.writeImageToDisk(
                [synth_img.clone()], [f"synth-{str(step_i)}.png"], output_dir
            )

            if save_pickle:
                for tensor, name in zip(
                    [
                        latent_in.detach().cpu().numpy(),
                        [n.detach().cpu().numpy() for n in noises],
                    ],
                    [f"w_latent-{str(step_i)}.pkl", f"noises-{str(step_i)}.pkl"],
                ):
                    with open(os.path.join(output_dir, name), "wb") as filestream:
                        pickle.dump(tensor, filestream)


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=str,
    help="Directory of images to invert",
)
@click.option(
    "--n-steps",
    default=2000,
    type=int,
    help="Number of optimization steps",
)
@click.option(
    "--optimize-extended-w-space/--no-optimize-extended-w-space",
    default=True,
    type=bool,
    help="Optimize W+ space? If False optimizes W space",
)
@click.option(
    "--output-dir",
    required=True,
    type=str,
    help="Directory to output inverted latent and reconstructed image to",
)
@click.option(
    "--save-pickle/--no-save-pickle",
    default=False,
    type=bool,
    help="Save pickles of latents, noises, and losses?",
)
@click.option(
    "--save-synth-every",
    default=500,
    type=int,
    help="Save synthesized image every N steps",
)
@click.pass_context
def invert_directory(
    ctx: click.Context,
    input_dir: str,
    n_steps: int,
    optimize_extended_w_space: bool,
    output_dir: str,
    save_pickle: bool,
    save_synth_every: int,
):
    """Invert entire directory of images"""
    for img_name in os.listdir(input_dir):
        ctx.invoke(
            gan_invert,
            image_path=os.path.join(input_dir, img_name),
            n_steps=n_steps,
            optimize_extended_w_space=optimize_extended_w_space,
            output_dir=os.path.join(output_dir, Path(img_name).stem),
            save_pickle=save_pickle,
            save_synth_every=save_synth_every,
        )


def _load_pickle(pickled_path):
    with open(pickled_path, "rb") as filestream:
        loaded_obj = pickle.load(filestream)
    return loaded_obj


@cli.command()
@click.option(
    "--gan-output-size",
    default=1024,
    type=int,
    help="Resolution of StyleGANv2 synthesized image output",
)
@click.option(
    "--noises-latent-path",
    default=os.path.join("data", "results", "noises-499.pkl"),
    type=str,
    help="Path to noises latent tensor",
)
@click.option(
    "--output-dir",
    default=os.path.join("data", "results"),
    type=str,
    help="Directory to output reconstructed image to",
)
@click.option(
    "--styleganv2-ckpt-path",
    default=os.path.join("checkpoints", "stylegan2-ffhq-config-f.pt"),
    type=str,
    help="Path to pretrained StyleGANv2 weights",
)
@click.option(
    "--w-latent-path",
    default=os.path.join("data", "results", "w_latent-499.pkl"),
    type=str,
    help="Path to W space latent tensor",
)
def reconstruct_from_latent(
    gan_output_size: int,
    noises_latent_path: str,
    output_dir: str,
    styleganv2_ckpt_path: str,
    w_latent_path: str,
):
    """Reconstructs synthesized image from latents"""
    os.makedirs(output_dir, exist_ok=True)

    styleganv2_gen = _setup_styleganv2(gan_output_size, styleganv2_ckpt_path)

    w_latent = _load_pickle(w_latent_path)
    w_latent = torch.from_numpy(w_latent).cuda()
    noises = _load_pickle(noises_latent_path)
    noises = [torch.from_numpy(n).cuda() for n in noises]

    # NOTE(brendan): in case w_latent is from the W space not the extended W+
    # space, repeat it
    if w_latent.shape[1] == 1:
        w_latent = w_latent.repeat(1, styleganv2_gen.n_latent, 1)

    synth_img, _ = styleganv2_gen([w_latent], input_is_latent=True, noise=noises)
    image_utils.writeImageToDisk([synth_img.clone()], ["reconstructed.png"], output_dir)


if __name__ == "__main__":
    cli()
