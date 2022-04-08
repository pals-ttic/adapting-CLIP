"""Implementation of GPU-based brute force box search."""

import numpy as np
import torch
import torch.nn.functional as F


class BruteForceBoxSearch():
    """Use pytorch to speed up. If matrix is too large, it may be out of memory.
    """

    def __init__(self, downsample=8):
        self.downsample = downsample

    def __call__(self, matrix, objective_cls):
        h, w = matrix.shape[:2]
        matrix = torch.from_numpy(matrix).cuda()
        # get new size
        self.h = h // self.downsample
        self.w = w // self.downsample
        # downsample matrix
        self.matrix = F.interpolate(
            matrix[None, None], (self.h, self.w), mode='bilinear')[0, 0]
        # get objective object
        self.objective = objective_cls(self.matrix)
        # get full intervals
        intervals = [[0, self.w-1], [0, self.h-1],
                     [0, self.w-1], [0, self.h-1]]
        # get coarse guess
        anchor_box = self.search(intervals)
        # rescale to original resolution
        anchor_box *= self.downsample
        x1, y1, x2, y2 = anchor_box
        # offset of adjustment
        offset_w = offset_h = self.downsample
        # back to original matrix
        self.matrix = matrix
        self.h, self.w = h, w
        # get new objective
        self.objective = objective_cls(self.matrix)
        # set intervals
        intervals = [
            [max(0, x1-offset_w), min(x1+offset_w, self.w-1)],
            [max(0, y1-offset_h), min(y1+offset_h, self.h-1)],
            [max(0, x2-offset_w), min(x2+offset_w, self.w-1)],
            [max(0, y2-offset_h), min(y2+offset_h, self.h-1)],
        ]
        # search box
        box = self.search(intervals)
        return box

    def search(self, intervals):
        # intervals is like [[x1_min, x1_max], [y1_min, y1_max], ...]
        x1 = torch.arange(intervals[0][0], intervals[0][1]+1).cuda()
        y1 = torch.arange(intervals[1][0], intervals[1][1]+1).cuda()
        x2 = torch.arange(intervals[2][0], intervals[2][1]+1).cuda()
        y2 = torch.arange(intervals[3][0], intervals[3][1]+1).cuda()
        boxes = torch.cartesian_prod(x1, y1, x2, y2)
        x1, y1, x2, y2 = boxes.transpose(0, 1)
        boxes = boxes[(x1 >= 0) & (y1 >= 0) & (x2 < self.w) &
                      (y2 < self.h) & (x2 > x1) & (y2 > y1)]
        box = boxes[self.objective.eval(boxes).argmax()]
        box = box.cpu().numpy()
        return box


class SumAreaObjective():
    """f(x) = (sum inside x) - alpha * (normalized area of x)
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, matrix):
        # precompute things
        self.matrix = matrix
        self.h, self.w = matrix.size(0), matrix.size(1)
        # for normalizing area
        self.area = float(self.h * self.w)
        # assume matrix is all positive
        self.sums = self.matrix.cumsum(1).cumsum(0)
        # pad the sums on top and left to deal with boundary cases
        self.sums = F.pad(self.sums, (1, 0, 1, 0))
        # get total sum for computing fraction
        self.total_sum = self.matrix.sum().item()
        return self

    def eval(self, boxes):
        frac = self._compute_frac(boxes)
        area = self._compute_area(boxes)
        return frac - self.alpha * area

    def _compute_frac(self, boxes):
        # boxes is Nx4, each is [x1, y1, x2, y2]
        x1, y1, x2, y2 = boxes.transpose(0, 1)
        # assume all boxes are valid
        bottom_right = self.sums[y2+1, x2+1]
        bottom_left = self.sums[y2+1, x1]
        top_right = self.sums[y1, x2+1]
        top_left = self.sums[y1, x1]
        box_sum = bottom_right - bottom_left - top_right + top_left
        return box_sum / (self.total_sum + 1e-8)

    def _compute_area(self, boxes):
        # boxes is Nx4, each is [x1, y1, x2, y2]
        x1, y1, x2, y2 = boxes.transpose(0, 1)
        # assume all boxes are valid
        return (x2 - x1 + 1).float() * (y2 - y1 + 1).float() / self.area


class FractionAreaObjective():
    """f(x) = (fraction of sum inside x) - alpha * (normalized area of x)
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, matrix):
        # precompute things
        self.matrix = matrix
        self.h, self.w = matrix.size(0), matrix.size(1)
        # for normalizing area
        self.area = float(self.h * self.w)
        # assume matrix is all positive
        self.sums = self.matrix.cumsum(1).cumsum(0)
        # pad the sums on top and left to deal with boundary cases
        self.sums = F.pad(self.sums, (1, 0, 1, 0))
        # get total sum for computing fraction
        self.total_sum = self.matrix.sum().item()
        return self

    def eval(self, boxes):
        frac = self._compute_frac(boxes)
        area = self._compute_area(boxes)
        return frac - self.alpha * area

    def _compute_frac(self, boxes):
        # boxes is Nx4, each is [x1, y1, x2, y2]
        x1, y1, x2, y2 = boxes.transpose(0, 1)
        # assume all boxes are valid
        bottom_right = self.sums[y2+1, x2+1]
        bottom_left = self.sums[y2+1, x1]
        top_right = self.sums[y1, x2+1]
        top_left = self.sums[y1, x1]
        box_sum = bottom_right - bottom_left - top_right + top_left
        return box_sum / (self.total_sum + 1e-8)

    def _compute_area(self, boxes):
        # boxes is Nx4, each is [x1, y1, x2, y2]
        x1, y1, x2, y2 = boxes.transpose(0, 1)
        # assume all boxes are valid
        return (x2 - x1 + 1).float() * (y2 - y1 + 1).float() / self.area


if __name__ == '__main__':
    import time
    h, w = 256, 256
    block_size = 32
    x1 = int(np.random.rand() * (w-block_size-1))
    y1 = int(np.random.rand() * (h-block_size-1))
    x2 = x1 + block_size
    y2 = y1 + block_size
    heatmap = np.zeros((h, w))
    heatmap[y1:y2+1, x1:x2+1] = 1.

    torch.cuda.synchronize()  # avoid counting cuda intialization time
    bf = BruteForceBoxSearch()
    objective = FractionAreaObjective(alpha=1.)
    start = time.time()
    print('BF Box: {}'.format(bf(heatmap, objective)))
    print('BF Time: %s' % (time.time()-start))
    print('-------------')
    print('GT Box: %s' % ([x1, y1, x2, y2]))
