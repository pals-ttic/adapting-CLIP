import numpy as np
import torch
from PIL import Image
import clip
from models.slic_vit import SLICViT


class ResNetHighRes(SLICViT):
    def __init__(self, **args):
        for key in args:
            if key not in ['model', 'high_res', 'alpha', 'temperature']:
                raise Exception(
                    '"{}" is not a valid argument to this model'.format(key))
        assert 'model' in args
        super().__init__(**args)

    def get_heatmap(self, im, text, resize=True):
        with torch.no_grad():
            # im is uint8 numpy
            h, w = im.shape[:2]
            im = Image.fromarray(im).convert('RGB')
            if resize:
                im = im.resize((224, 224))
            im = self.model.preprocess(im).unsqueeze(0).cuda()

            image_features = self.model(im, masks=None)  # 1xCxHxW

            text = clip.tokenize([text]).cuda()
            text_features = self.model.encode_text(text)

            image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

            heatmap = (image_features * text_features[:, :, None, None]).sum(1)
            heatmap = heatmap.cpu().float().numpy()[0]

            heatmap = np.exp(heatmap / self.temperature)

        return heatmap
