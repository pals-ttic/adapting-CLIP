import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import spatial_clip
import clip


class CLIPSpatialResNet(nn.Module):
    def __init__(self, model='RN50', high_res=True):
        super(CLIPSpatialResNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = spatial_clip.load(
            model, device=device, high_res=high_res)
        self.high_res = high_res

    def encode_text(self, text):
        return self.model.encode_text(text)

    def encode_image(self, image):
        return self.model.encode_image(image)

    def forward(self, im, masks=None):
        with torch.no_grad():
            im = im.type(self.model.dtype)
            # pad image to remove boundary effect
            pad = 64  # heuristic pad size
            pad = (pad, pad, pad, pad)
            padded_im = F.pad(im, pad, 'constant', 0)
            # get features
            features = self.model.encode_image(padded_im)
            # crop out center to remove pad
            if self.high_res:
                target_size_h, target_size_w = im.size(-2), im.size(-1)
            else:
                target_size_h, target_size_w = im.size(-2)//32, im.size(-1)//32
            # compute new pad size
            pad_h = (features.size(-2) - target_size_h) // 2
            pad_w = (features.size(-1) - target_size_w) // 2
            features = features[:, :, pad_h:pad_h+target_size_h, pad_w:pad_w+target_size_w]
            # interpolate back to 224x224, use nearest to reproduce denseclip
            features = F.upsample(features, size=(
                im.size(-2), im.size(-1)), mode='bilinear', align_corners=None)  # 1xCxHxW
            assert features.size(0) == 1
            if masks is None:  # return per-pixel features if no masks provided
                return features  # 1xCxHxW
            features = features[0].permute(1, 2, 0)  # HxWxC
            mask_features = []
            for mask in masks:
                mask_features.append(features[mask].mean(0))
            mask_features = torch.stack(mask_features, 0)
            mask_features = mask_features[None]  # add dummy batch dim
        return mask_features
