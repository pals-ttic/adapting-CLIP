import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import linear, _in_projection_packed
import clip


class MaskCLIPViT(nn.Module):
    def __init__(self, name):
        super().__init__()
        assert name.startswith('ViT-')
        self.model, self.preprocess = clip.load(name)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def encode_image(self, x):
        visual = self.model.visual
        x = visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        outs = []
        transformer = visual.transformer
        for resblock in transformer.resblocks:
            x = resblock(x)
            outs.append(x)
        # x = visual.transformer(x)
        x = outs[-2]
        resblock = transformer.resblocks[-1]
        attn = resblock.attn
        x = x + resblock.ln_1(x)
        q, k, v = _in_projection_packed(x, x, x, attn.in_proj_weight, attn.in_proj_bias)
        v = linear(v.transpose(0, 1).contiguous(), attn.out_proj.weight, attn.out_proj.bias)
        v = v + resblock.mlp(resblock.ln_2(v))
        v = v[:, 1:] # N(L-1)D

        # x = x.permute(1, 0, 2)  # LND -> NLD

        # x = visual.ln_post(x[:, 0, :])

        # if visual.proj is not None:
            # x = x @ visual.proj
        v = visual.ln_post(v)
        if visual.proj is not None:
            v = v @ visual.proj

        v = v.view(v.size(0), int(np.sqrt(v.size(1))), int(np.sqrt(v.size(1))), -1)
        return v

    def forward(self, im, text):
        image_features = self.encode_image(im)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # im = (np.random.rand(256, 334, 3) * 255).astype(np.uint8)
    im = Image.open(os.path.join(dir_path, 'joker_batman.jpg'))
    plt.figure();plt.imshow(im); plt.savefig(os.path.join(dir_path, 'original.png'))
    model = MaskCLIPViT('ViT-L/14').cuda()
    with torch.no_grad():
        im = model.preprocess(im).unsqueeze(0).half().cuda()
        text = clip.tokenize(['joker', 'batman']).cuda()
        logits, image_features, text_features = model(im, text)
        probs = torch.softmax(logits.squeeze(), -1)
        plt.figure();plt.imshow(probs[..., 0].float().cpu().numpy()); plt.savefig(os.path.join(dir_path, 'joker_heatmap.png'))
        plt.figure();plt.imshow(probs[..., 1].float().cpu().numpy()); plt.savefig(os.path.join(dir_path, 'batman_heatmap.png'))
