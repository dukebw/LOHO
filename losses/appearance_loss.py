import torch
import torch.nn as nn
from torch.nn import functional as F

import os

from losses.custom_loss import custom_loss, prepare_mask
from networks.vgg_activations import VGG16_Activations, VGG19_Activations, Vgg_face_dag


class AppearanceLoss(nn.Module):
    def __init__(self, VGG16_ACTIVATIONS_LIST=[21], normalize=False, distance="l2"):

        super(AppearanceLoss, self).__init__()

        self.vgg16_act = VGG16_Activations(VGG16_ACTIVATIONS_LIST)
        self.vgg16_act.eval()

        self.normalize = normalize
        self.distance = distance

    def get_features(self, model, x):

        return model(x)

    def mask_features(self, x, mask):

        mask = prepare_mask(x, mask)
        return x * mask, mask

    def cal_appearance(self, model, x, x_hat, mask1=None, mask2=None):
        # Get features from the model for x and x_hat
        with torch.no_grad():
            act_x = self.get_features(model, x)
        for layer in range(0, len(act_x)):
            act_x[layer].detach_()

        act_x_hat = self.get_features(model, x_hat)

        loss = 0.0
        for layer in range(0, len(act_x)):

            # mask features if present
            if mask1 is not None:
                feat_x, mask_x = self.mask_features(act_x[layer], mask1)
                on_pixels = torch.sum(mask_x[0, 0, ...])
                appearance_x = torch.sum(feat_x, axis=[2, 3]) / (on_pixels + 1e-8)
            else:
                feat_x = act_x[layer]
                appearance_x = torch.mean(feat_x, axis=[2, 3])

            if mask2 is not None:
                feat_x_hat, mask_x_hat = self.mask_features(act_x_hat[layer], mask2)
                on_pixels = torch.sum(mask_x_hat[0, 0, ...])
                appearance_x_hat = torch.sum(feat_x_hat, axis=[2, 3]) / (
                    on_pixels + 1e-8
                )
            else:
                feat_x_hat = act_x_hat[layer]
                appearance_x_hat = torch.mean(feat_x_hat, axis=[2, 3])

            # compute layer wise loss and aggregate
            loss += custom_loss(
                appearance_x,
                appearance_x_hat,
                mask=None,
                loss_type=self.distance,
                include_bkgd=True,
            )

        loss = loss / len(act_x)

        return loss

    def forward(self, x, x_hat, mask1=None, mask2=None):
        x = x.cuda()
        x_hat = x_hat.cuda()

        # resize images to 256px resolution
        N, C, H, W = x.shape
        upsample2d = nn.Upsample(
            scale_factor=256 / H, mode="bilinear", align_corners=True
        )

        x = upsample2d(x)
        x_hat = upsample2d(x_hat)

        loss = self.cal_appearance(self.vgg16_act, x, x_hat, mask1=mask1, mask2=mask2)

        return loss
