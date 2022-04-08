import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPMaskedSpatialViT(nn.Module):
    def __init__(self, patch_size=16, upsample=1, start_block=0, align_corners=None):
        super(CLIPMaskedSpatialViT, self).__init__()

        assert patch_size == 14 or patch_size == 16 or patch_size == 32
        self.patch_size = patch_size
        if self.patch_size == 14:
            self.target_size = 16
        elif self.patch_size == 16:
            self.target_size = 14
        else:
            self.target_size = 7
        if self.patch_size == 14:
            self.model, self.preprocess = clip.load(
                "ViT-L/{}".format(self.patch_size))
        else:
            self.model, self.preprocess = clip.load(
                "ViT-B/{}".format(self.patch_size))

        self.align_corners = align_corners

        assert (upsample == 1) or (upsample & (upsample-1) == 0)  # power of 2
        self.upsample = upsample
        self.target_size = self.target_size * self.upsample
        self.stem_stride = self.patch_size // upsample
        self.model.visual.conv1.stride = self.stem_stride
        self.model.visual.conv1.padding = (
            self.patch_size - 1) // 2  # TODO: make it more precise
        self.model.visual.positional_embedding = self.upsample_pos_emb(
            self.model.visual.positional_embedding)

        self.start_block = start_block

    def upsample_pos_emb(self, emb):
        # upsample the pretrained embedding for higher resolution
        # emb size NxD
        first = emb[:1, :]
        emb = emb[1:, :]
        N, D = emb.size(0), emb.size(1)
        size = int(np.sqrt(N))
        assert size * size == N
        new_size = size * self.upsample
        emb = emb.permute(1, 0)
        emb = emb.view(1, D, size, size).contiguous()
        emb = F.upsample(emb, size=new_size, mode='bilinear',
                         align_corners=self.align_corners)
        emb = emb.view(D, -1).contiguous()
        emb = emb.permute(1, 0)
        emb = torch.cat([first, emb], 0)
        emb = nn.parameter.Parameter(emb.half())
        return emb

    def masks_to_attn_map(self, masks):
        # masks size NxHxW
        N = masks.size(0)
        # masks is 1 for the object and 0 for others, need to invert it
        masks = 1 - masks.bool().float()
        # interpolate to target size
        masks = masks.unsqueeze(1).float()
        target_size = (self.target_size, self.target_size)
        masks = F.interpolate(masks, size=target_size,
                              mode='nearest', align_corners=None)
        masks = masks.squeeze(1)
        attn_map = masks.view(N, -1)
        attn_map = torch.cat([attn_map, 1-torch.eye(N).to(attn_map.device)], 1)
        attn_map = attn_map.bool().half() * (-100)
        return attn_map

    def encode_text(self, text):
        return self.model.encode_text(text)

    def encode_image(self, image):
        return self.model.encode_image(image)

    def forward(self, im, masks):
        vit = self.model.visual
        x = im.type(self.model.dtype)

        x = vit.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        attn_mask = self.masks_to_attn_map(masks)
        attn_mask = attn_mask.type(self.model.dtype)
        num_masks = attn_mask.size(0)
        for block_idx, resblock in enumerate(vit.transformer.resblocks):
            if block_idx == self.start_block:
                gv = x[:1]
                gv = gv.repeat(num_masks, 1, 1)  # LND
            if block_idx >= self.start_block:
                attn = resblock.attn
                source = resblock.ln_1(torch.cat([x[1:], gv], 0))
                gv = gv + attn(
                    source[-num_masks:],
                    source,
                    source,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
                gv = gv + resblock.mlp(resblock.ln_2(gv))
            x = resblock(x)

        gv = gv.permute(1, 0, 2)
        gv = vit.ln_post(gv)
        if vit.proj is not None:
            gv = (gv.view(-1, gv.size(-1)) @
                  vit.proj).view(gv.size(0), gv.size(1), -1)

        return gv
