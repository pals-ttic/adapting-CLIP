"""Implements ZGS Evaluation."""

import ast
import torch
import numpy as np

from utils.bbox_tools import matched_bbox_iou


class GroundingEvaluator(object):
    def __init__(self, gt_dataset, iou_thresh=0.5):
        """Grounding Evaluation given GT dataset.

        Args:
            gt_dataset ([ZSG Dataset]): A supported dataset in data.zsg
            iou_thresh (float, optional): IoU threshold to compute accurancy. Defaults to 0.5.
        """
        self.gt_dataset = gt_dataset
        self.iou_thres = iou_thresh

    def __call__(self, pred_bboxes):
        """[summary]

        Args:
            predictions ([Tensor]): Batch x 4 (y1x1y2x2) format.
        """
        gt_bboxes = []
        for idx in range(len(self.gt_dataset)):
            gt_bboxes.append(ast.literal_eval(self.gt_dataset.bboxes[idx]))
        gt_bboxes = torch.from_numpy(np.array(gt_bboxes))

        iou = matched_bbox_iou(pred_bboxes, gt_bboxes)
        acc = torch.mean((iou > self.iou_thres).double())
        return acc
