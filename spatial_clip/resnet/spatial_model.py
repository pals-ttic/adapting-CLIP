"""Implements Spatial CLIP Model.

Modified https://github.com/openai/CLIP/blob/main/clip/model.py
to support spatial prediction.
"""

from collections import OrderedDict
from typing import Tuple, Union

import torch

from clip import tokenize
from clip.model import CLIP, convert_weights, ModifiedResNet
from .dilated_model import ModifiedSpatialResNetDilated, AttentionSpatial2d


class ModifiedSpatialResNet(ModifiedResNet):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)
        # Override the Self Attention Pooling to Spatial Prediction.
        embed_dim = width * 32
        self.attnpool = AttentionSpatial2d(
            input_resolution // 32, embed_dim, heads, output_dim)


class Spatial_CLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # spatial vision
                 high_res: bool,
                 neighbor_window=None,
                 split_layer=-1,
                 ):
        super().__init__(embed_dim, image_resolution, vision_layers, vision_width,
                         vision_patch_size, context_length, vocab_size,
                         transformer_width, transformer_heads, transformer_layers)
        self.high_res = high_res
        self.neighbor_window = neighbor_window
        # Override the visual branch.
        if isinstance(vision_layers, (tuple, list)):
            # Should not be used for ResNet Model.
            assert neighbor_window is None
            vision_heads = vision_width * 32 // 64
            visual_block = ModifiedSpatialResNetDilated if high_res else ModifiedSpatialResNet
            self.visual = visual_block(layers=vision_layers,
                                       output_dim=embed_dim,
                                       heads=vision_heads,
                                       input_resolution=image_resolution,
                                       width=vision_width)
        else:
            raise ValueError('Only ResNet based CLIP is supported.')

    # Override CLIP Forward to perform per-pixel simiarlity.
    def forward(self, image, text_list):
        image_features = self.encode_image(image)
        text = tokenize(text_list).to(image.device)
        text_features = self.encode_text(text).type(self.dtype)

        # normalized features
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # m: image batch size; n: num of text prompts
        spatial_logits = logit_scale * \
            torch.einsum("mchw, nc -> mnhw", image_features, text_features)
        return spatial_logits


def build_spatial_model(state_dict: dict, high_res: bool = False, neighbor_window=None, split_layer=-1):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith(
            "visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(
            f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + \
            1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = Spatial_CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        high_res=high_res, neighbor_window=neighbor_window, split_layer=split_layer
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    # False for the average filter layer.
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
