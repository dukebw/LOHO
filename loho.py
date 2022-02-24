import argparse
import math
import os
import cv2
import numpy as np
import pickle
from PIL import Image

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision import utils
from tqdm import tqdm

from networks import lpips
from networks import deeplab_xception_transfer
from networks.style_gan_2 import Generator
from networks.graphonomy_inference import get_mask
from losses.style_loss import StyleLoss
from losses.appearance_loss import AppearanceLoss
from losses.noise_loss import noise_regularize, noise_normalize_
from datasets.ffhq import process_image, dilate_erosion_mask
from utils import optimizer_utils, image_utils


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--save_synth_every", type=int, default=500)
    parser.add_argument("--save_pickle", type=int, default=0)
    parser.add_argument("--noise_regularize", type=float, default=1e5)
    parser.add_argument("--image1", type=str, default="00018.png")
    parser.add_argument("--image2", type=str, default="00200.png")
    parser.add_argument("--image3", type=str, default="00079.png")
    parser.add_argument("--use_GO", type=int, default=1, help="Use GO or no-GO")
    parser.add_argument(
        "--style_mask_type", type=int, default=1, help="1.ori, 2.comp. mask"
    )
    parser.add_argument("--lpips_vgg_blocks", type=str, default="4,5")
    parser.add_argument("--style_vgg_layers", type=str, default="3,8,15,22")
    parser.add_argument("--appearance_vgg_layers", type=str, default="1")
    parser.add_argument("--lambda_facerec", type=float, default=1.0)
    parser.add_argument("--lambda_hairstyle", type=float, default=15000.0)
    parser.add_argument("--lambda_hairappearance", type=float, default=40.0)
    parser.add_argument("--lambda_hairrec", type=float, default=1.0)

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, 256)

    ############################## PREPARE DATA

    # Define paths to tuples and checkpoints
    raw = "data/images"
    mask = "data/masks"
    background = "data/backgrounds"
    softmask = "data/softmasks"
    input_name = (
        args.image1.split(".")[0]
        + "_"
        + args.image2.split(".")[0]
        + "_"
        + args.image3.split(".")[0]
    )
    dest = os.path.join("data/results", input_name)
    if not os.path.exists(dest):
        os.makedirs(dest)
    styleganv2_ckpt_path = "checkpoints/stylegan2-ffhq-config-f.pt"
    graphonomy_model_path = "checkpoints/inference.pth"

    # Get path to image files
    image_files = image_utils.getImagePaths(
        raw, mask, background, args.image1, args.image2, args.image3
    )

    # Get images and masks
    I_1, M_1, HM_1, H_1, FM_1, F_1, FG_1 = process_image(
        image_files["I_1_path"], image_files["M_1_path"], size=resize, normalize=1
    )
    I_2, M_2, HM_2, H_2, FM_2, F_2, FG_2 = process_image(
        image_files["I_2_path"], image_files["M_2_path"], size=resize, normalize=1
    )
    I_3, M_3, HM_3, H_3, FM_3, F_3, FG_3 = process_image(
        image_files["I_3_path"], image_files["M_3_path"], size=resize, normalize=1
    )

    # Make cuda
    I_1, M_1, HM_1, H_1, FM_1, F_1, FG_1 = optimizer_utils.make_cuda(
        [I_1, M_1, HM_1, H_1, FM_1, F_1, FG_1]
    )
    I_2, M_2, HM_2, H_2, FM_2, F_2, FG_2 = optimizer_utils.make_cuda(
        [I_2, M_2, HM_2, H_2, FM_2, F_2, FG_2]
    )
    I_3, M_3, HM_3, H_3, FM_3, F_3, FG_3 = optimizer_utils.make_cuda(
        [I_3, M_3, HM_3, H_3, FM_3, F_3, FG_3]
    )

    # Expand batch dim
    I_1, M_1, HM_1, H_1, FM_1, F_1, FG_1 = image_utils.addBatchDim(
        [I_1, M_1, HM_1, H_1, FM_1, F_1, FG_1]
    )
    I_2, M_2, HM_2, H_2, FM_2, F_2, FG_2 = image_utils.addBatchDim(
        [I_2, M_2, HM_2, H_2, FM_2, F_2, FG_2]
    )
    I_3, M_3, HM_3, H_3 = image_utils.addBatchDim([I_3, M_3, HM_3, H_3])

    HM_2D, HM_2E = dilate_erosion_mask(image_files["M_2_path"], resize)
    HM_2D, HM_2E = HM_2D.float(), HM_2E.float()
    HM_2D, HM_2E = optimizer_utils.make_cuda([HM_2D, HM_2E])
    HM_2D, HM_2E = image_utils.addBatchDim([HM_2D, HM_2E])
    ignore_region = HM_2D - HM_2E
    mask_loss_region = 1 - ignore_region

    # Write masks to disk for visualization
    image_utils.writeMaskToDisk(
        [HM_1, HM_2, HM_2D, HM_2E, ignore_region],
        ["HM_1.png", "HM_2.png", "HM_2D.png", "HM_2E.png", "ignore_region.png"],
        dest,
    )
    image_utils.writeImageToDisk(
        [I_1.clone(), I_2.clone(), I_3.clone()], ["I_1.png", "I_2.png", "I_3.png"], dest
    )

    ############################# LOAD MODELS

    lpips_vgg_blocks = args.lpips_vgg_blocks.split(",")
    # LPIPS MODEL
    face_percept = lpips.PerceptualLoss(
        model="net-lin",
        net="vgg",
        vgg_blocks=["1", "2", "3", "4", "5"],
        use_gpu=device.startswith("cuda"),
    )
    hair_percept = lpips.PerceptualLoss(
        model="net-lin",
        net="vgg",
        vgg_blocks=lpips_vgg_blocks,
        use_gpu=device.startswith("cuda"),
    )

    # STYLE + APPEARANCE MODEL
    style_vgg_layers = args.style_vgg_layers.split(",")
    style_vgg_layers = [int(i) for i in style_vgg_layers]
    style = StyleLoss(
        distance="l2", VGG16_ACTIVATIONS_LIST=style_vgg_layers, normalize=False
    )
    style.cuda()

    appearance_vgg_layers = args.appearance_vgg_layers.split(",")
    appearance_vgg_layers = [int(i) for i in appearance_vgg_layers]
    appearance = AppearanceLoss(
        distance="l2", VGG16_ACTIVATIONS_LIST=appearance_vgg_layers, normalize=False
    )
    appearance.cuda()

    # GRAPHONOMY MODEL
    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
        n_classes=20,
        hidden_layers=128,
        source_classes=7,
    )

    state_dict = torch.load(graphonomy_model_path)
    net.load_source_model(state_dict)
    net.cuda()
    net.eval()

    # GENERATOR
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(styleganv2_ckpt_path)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(I_1.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(I_1.shape[0], 1)
    # copy over to W+
    latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []

    losses_log = {
        "facerec": [],
        "hairstyle_3g": [],
        "hairstyle_2g": [],
        "hairappearance_3g": [],
        "hairappearance_2g": [],
        "hairrec": [],
        "noise": [],
    }
    g_S2_vector_norms = {i: [] for i in range(latent_in.shape[1])}
    g_S3_vector_norms = {i: [] for i in range(latent_in.shape[1])}
    g_L_vector_norms = {i: [] for i in range(latent_in.shape[1])}
    g_L_hat_vector_norms = {i: [] for i in range(latent_in.shape[1])}
    dot_gL_gS2 = {i: [] for i in range(latent_in.shape[1])}
    dot_gLhat_gS2 = {i: [] for i in range(latent_in.shape[1])}

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        I_G, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = I_G.shape

        if height > 256:
            factor = height // 256

            I_G = I_G.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            I_G = I_G.mean([3, 5])

        # get hair mask of synthesized image
        predictions, HM_G, FM_G = get_mask(net, I_G)
        HM_G = HM_G.float()
        HM_G = torch.unsqueeze(HM_G, axis=0)
        FM_G = torch.unsqueeze(FM_G, axis=0)

        # LPIPS on face
        target_mask = FM_1 * (1 - HM_2D)
        facerec_loss = args.lambda_facerec * face_percept(I_G, I_1, mask=target_mask)
        losses_log["facerec"].append(facerec_loss.item())

        # LPIPS on hair
        hairrec_loss = args.lambda_hairrec * hair_percept(I_G, I_2, HM_2E)
        losses_log["hairrec"].append(hairrec_loss.item())

        # Style Loss on hair
        # compute target mask on synthesized image
        if i < 1000:
            if args.style_mask_type == 1:
                mask2 = HM_G
            elif args.style_mask_type == 2:
                mask2 = HM_2 + (ignore_region * HM_G)
                mask2 = torch.where(
                    mask2 >= 1, torch.ones_like(mask2), torch.zeros_like(mask2)
                )
        H_G = I_G * mask2

        # style loss between H_2 and H_G
        hairstyle_2g_loss = args.lambda_hairstyle * style(
            H_2, H_G, mask1=HM_2, mask2=mask2
        )
        hairappearance_2g_loss = args.lambda_hairappearance * appearance(
            H_2, H_G, mask1=HM_2, mask2=mask2
        )

        # style loss between H_3 and H_G
        hairstyle_3g_loss = args.lambda_hairstyle * style(
            H_3, H_G, mask1=HM_3, mask2=mask2
        )
        hairappearance_3g_loss = args.lambda_hairappearance * appearance(
            H_3, H_G, mask1=HM_3, mask2=mask2
        )

        losses_log["hairstyle_2g"].append(hairstyle_2g_loss.item())
        losses_log["hairstyle_3g"].append(hairstyle_3g_loss.item())
        losses_log["hairappearance_2g"].append(hairappearance_2g_loss.item())
        losses_log["hairappearance_3g"].append(hairappearance_3g_loss.item())

        n_loss = args.noise_regularize * noise_regularize(noises)
        losses_log["noise"].append(n_loss.item())

        loss = facerec_loss + n_loss

        if i < 1000:
            loss += hairrec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss += hairstyle_3g_loss + hairappearance_3g_loss

            if args.use_GO == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                # Accumulate gradients from losses that do not participate in projection loss
                # NOTE: Use clone to get .grad since it is referencing memory location

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                simple_L = latent_in.grad.clone()
                simple_noises = [n.grad.clone() for n in noises]

                # Get gradient g_L
                optimizer.zero_grad()
                hairrec_loss.backward(retain_graph=True)
                g_L = latent_in.grad.clone()
                g_L_noises = [n.grad.clone() for n in noises]

                # Get gradient g_S_2
                optimizer.zero_grad()
                (hairstyle_2g_loss + hairappearance_2g_loss).backward(retain_graph=True)
                g_S2 = latent_in.grad.clone()
                g_S2_noises = [n.grad.clone() for n in noises]

                # Get gradient g_S_3
                optimizer.zero_grad()
                (hairstyle_3g_loss + hairappearance_3g_loss).backward(retain_graph=True)
                g_S3 = latent_in.grad.clone()

                # Compute gradient orthogonalization
                # g_L, g_S_2  are [1, 18, 512] matrices
                # <g_L, g_S_2 + > will do dot between <g_L[1, i, 512], g_S_2[1, i, 512]>

                # The following code does <g_L, g_S_2> / <g_S_2, g_S_2>
                norm_vector = []
                for w_pos in range(len(g_L[0])):
                    g_S2_hat = F.normalize(g_S2[0, w_pos, :].unsqueeze(0), p=2)
                    dot_L_S2 = (
                        torch.dot(g_L[0, w_pos, :], g_S2_hat.squeeze()) * g_S2_hat
                    )
                    norm_vector.append(dot_L_S2)

                norm_vector = torch.stack(norm_vector)  # [18 x 1 x 512]
                norm_vector = torch.transpose(norm_vector, 0, 1)  # [1 x 18 x 512]

                adjusted_g_L = g_L - norm_vector

                # Record gradient norms
                for idx in range(len(g_L[0])):
                    g_S2_vector_norms[idx].append(torch.norm(g_S2[0, idx, :]).item())
                    g_S3_vector_norms[idx].append(torch.norm(g_S3[0, idx, :]).item())
                    g_L_vector_norms[idx].append(torch.norm(g_L[0, idx, :]).item())
                    g_L_hat_vector_norms[idx].append(
                        torch.norm(adjusted_g_L[0, idx, :]).item()
                    )
                    dot_gL_gS2[idx].append(
                        torch.dot(g_L[0, idx, :], g_S2[0, idx, :]).item()
                    )
                    dot_gLhat_gS2[idx].append(
                        torch.dot(adjusted_g_L[0, idx, :], g_S2[0, idx, :]).item()
                    )

                # Do update
                optimizer.zero_grad()
                loss = (
                    facerec_loss
                    + hairstyle_3g_loss
                    + hairappearance_3g_loss
                    + hairrec_loss
                    + n_loss
                )
                loss.backward()

                # assign precomputed statistics to gradient parameter
                latent_in.grad = simple_L + adjusted_g_L

                for idx in range(len(noises)):
                    noises[idx].grad = noises[idx].grad - g_L_noises[idx]

                optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perc_face: {facerec_loss.item():.4f};"
                f"perc_hair: {hairrec_loss.item():.4f};"
                f"style_hair: {hairstyle_3g_loss.item():.4f};"
                f"app_hair: {hairappearance_3g_loss.item():.4f};"
                f"noise regularize: {n_loss.item():.4f};"
            )
        )

        if (i + 1) % args.save_synth_every == 0:
            image_utils.writeImageToDisk([I_G.clone()], [f"synth-{str(i)}.png"], dest)
            image_utils.writeMaskToDisk(
                [HM_G, FM_G],
                [f"synth_hair_mask-{str(i)}.png", f"synth_face_mask-{str(i)}.png"],
                dest,
            )

    img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

    noise_single = []
    for noise in noises:
        noise_single.append(noise[0:1].detach().cpu().numpy())

    if args.save_pickle:
        image_utils.writePickleToDisk(
            [
                latent_in[0],
                noise_single,
                losses_log,
                g_S2_vector_norms,
                g_S3_vector_norms,
                g_L_vector_norms,
                g_L_hat_vector_norms,
                dot_gL_gS2,
                dot_gLhat_gS2,
            ],
            [
                "w_latent.pkl",
                "noises.pkl",
                "losses.pkl",
                "g_S2_vector_norms.pkl",
                "g_S3_vector_norms.pkl",
                "g_L_vector_norms.pkl",
                "g_L_hat_vector_norms.pkl",
                "dot_gL_gS2.pkl",
                "dot_gLhat_gS2.pkl",
            ],
            dest,
        )

    ########### INPAINT BACKGROUND

    # Get softmask
    with open(
        os.path.join(softmask, args.image1.split(".")[0] + ".pkl"), "rb"
    ) as handle:
        softmask = pickle.load(handle)

    # Get inpainted background
    background = cv2.imread(os.path.join(background, args.image1))
    background = cv2.resize(background, (512, 512))

    img_gen = image_utils.makeImage(img_gen)[0]  # in RGB
    img_gen = cv2.cvtColor(cv2.resize(img_gen, (512, 512)), cv2.COLOR_BGR2RGB)

    result = (softmask * img_gen) + (1 - softmask) * background
    result = result.astype(np.uint8)
    cv2.imwrite(os.path.join(dest, "result.png"), result)
