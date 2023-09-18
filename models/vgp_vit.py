import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic
from utils.box_search import BruteForceBoxSearch, FractionAreaObjective
import clip
from spatial_clip import CLIPMaskedSpatialViT
from spatial_clip import CLIPSpatialResNet


class VGPViT(nn.Module):
    def __init__(self, model='vit14', alpha=0.8, n_segments=[10, 50, 100, 200],
                 aggregation='mean', temperature=1., compactness=50,
                 sigma=0, **args):
        super().__init__()
        if model == 'vit14':
            args['patch_size'] = 14
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'vit16':
            args['patch_size'] = 16
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'vit32':
            args['patch_size'] = 32
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'RN50':
            self.model = CLIPSpatialResNet(**args)
        elif model == 'RN50x4':
            self.model = CLIPSpatialResNet(**args)
        else:
            raise Exception('Invalid model name: {}'.format(model))
        self.alpha = alpha
        self.n_segments = n_segments
        self.aggregation = aggregation
        self.temperature = temperature
        self.compactness = compactness
        self.sigma = sigma

    def get_masks(self, im):
        masks = []
        # Do SLIC with different number of segments so that it has a hierarchical scale structure
        # This can average out spurious activations that happens sometimes when the segments are too small
        for n in self.n_segments:
            segments_slic = slic(im.astype(
                np.float32)/255., n_segments=n, compactness=self.compactness, sigma=self.sigma)
            for i in np.unique(segments_slic):
                mask = segments_slic == i
                masks.append(mask)
        masks = np.stack(masks, 0)
        return masks

    def get_mask_scores(self, im, text):
        with torch.no_grad():
            # im is uint8 numpy
            h, w = im.shape[:2]
            im = Image.fromarray(im).convert('RGB')
            im = im.resize((224, 224))
            masks = self.get_masks(np.array(im))
            masks = torch.from_numpy(masks.astype(np.bool)).cuda()
            im = self.model.preprocess(im).unsqueeze(0).cuda()

            image_features = self.model(im, masks)
            image_features = image_features.permute(0, 2, 1)

            text = clip.tokenize([text]).cuda()
            text_features = self.model.encode_text(text)

            image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

            logits = (image_features * text_features.unsqueeze(-1)).sum(1)
            assert logits.size(0) == 1
            logits = logits.cpu().float().numpy()[0]

        return masks.cpu().numpy(), logits

    def get_heatmap(self, im, text):
        masks, logits = self.get_mask_scores(im, text)
        heatmap = list(np.nan + np.zeros(masks.shape, dtype=np.float32))
        for i in range(len(masks)):
            mask = masks[i]
            score = logits[i]
            heatmap[i][mask] = score
        heatmap = np.stack(heatmap, 0)

        heatmap = np.exp(heatmap / self.temperature)

        if self.aggregation == 'mean':
            heatmap = np.nanmean(heatmap, 0)
        elif self.aggregation == 'median':
            heatmap = np.nanmedian(heatmap, 0)
        elif self.aggregation == 'max':
            heatmap = np.nanmax(heatmap, 0)
        elif self.aggregation == 'min':
            heatmap = -np.nanmin(heatmap, 0)
        else:
            assert False

        mask_valid = np.logical_not(np.isnan(heatmap))
        _min = heatmap[mask_valid].min()
        _max = heatmap[mask_valid].max()
        heatmap[mask_valid] = (heatmap[mask_valid] -
                               _min) / (_max - _min + 1e-8)
        heatmap[np.logical_not(mask_valid)] = 0.
        return heatmap

    def forward(self, im, phrases, **args):
        # temporary override paramters in init
        _args = {key: getattr(self, key) for key in args}
        for key in args:
            setattr(self, key, args[key])
        # forward
        heatmaps = [self.get_heatmap(im, text) for text in phrases]
        # restore paramters
        for key in args:
            setattr(self, key, _args[key])
        return heatmaps
