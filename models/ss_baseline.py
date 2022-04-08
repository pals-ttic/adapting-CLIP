import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
from models.slic_vit import SLICViT
from utils.selective_search import selective_search


class SSBaseline(SLICViT):
    def __init__(self, num_proposals=100, scale=500, sigma=0.9, min_size=50, **args):
        if 'model' in args:
            assert args['model'].startswith('vit')
        super().__init__(**args)
        self.num_proposals = num_proposals
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size

    def get_boxes(self, im):
        _, regions = selective_search(
            im, scale=self.scale, sigma=self.sigma, min_size=self.min_size)
        boxes = np.array([list(x['rect']) for x in regions])
        boxes = boxes[:self.num_proposals]
        return boxes

    def forward(self, im, text, **args):
        # temporary override paramters in init
        _args = {key: getattr(self, key) for key in args}
        for key in args:
            setattr(self, key, args[key])
        # forward
        h, w = im.shape[:2]
        proposals = self.get_boxes(im)
        crops = []
        for box in proposals:
            x1, y1, x2, y2 = box
            crop = im[y1:y2+1, x1:x2+1]
            crop = Image.fromarray(crop).convert('RGB')
            crop = crop.resize((224, 224))
            crop = self.model.preprocess(crop).cuda()
            crops.append(crop)
        crops = torch.stack(crops, 0)
        with torch.no_grad():
            image_features = self.model.encode_image(crops)

            text = clip.tokenize([text]).cuda()
            text_features = self.model.encode_text(text)

            image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

            logits = (image_features * text_features).sum(1)  # [B, n_crops]
            logits = logits.cpu().float().numpy()
            sorted_idx = np.argsort(logits)[::-1]

        boxes_pred = proposals[sorted_idx]

        # restore paramters
        for key in args:
            setattr(self, key, _args[key])

        # heatmap is None
        return boxes_pred, None
