"""Bounding box tools."""

import torch


def unnormalize_bbox(normalized_box, image_size):
    """Unormalize a batch of bbox given image size.

    Args:
        normalized_box (Tensor): Batch x 4 (y1x1y2x2) format.
        image_size ([type]): Batch x 2 (height x width) format.

    Returns:
        [type]: [description]
    """
    normalized_box[:, :2] = image_size * normalized_box[:, :2]
    normalized_box[:, 2:] = image_size * normalized_box[:, 2:]
    return normalized_box


def matched_bbox_iou(bbox1, bbox2):
    """Computes Bounding Box Intersection over Union.
    (Modified from detetron2.)

    Args:
        bbox1 (Tensor): Batch x 4 (y1x1y2x2) format.
        bbox2 (Tensor): Batch x 4 (y1x1y2x2) format.
    """
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    lt = torch.max(bbox1[:, :2], bbox2[:, :2])  # [N,2]
    rb = torch.min(bbox1[:, 2:], bbox2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou
