import argparse
import os.path as osp
from tqdm import tqdm
import numpy as np
import torch
import csv
from models.vgp_vit import VGPViT
from models.slic_vit import SLICViT
from models.ss_baseline import SSBaseline
from models.resnet_high_res import ResNetHighRes
from utils.zsg_data import FlickrDataset, VGDataset
from utils.vgp_data import FlickrVGPsDataset
from utils.grounding_evaluator import GroundingEvaluator, GroudingEvaluatorVGPs
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.simplefilter('ignore')

def eval(model, dataset, iou_thr):
    pred = []
    for idx in tqdm(range(len(dataset))):
        im = dataset[idx]['image']
        text = dataset[idx]['phrases'][0]
        bbox_pred, _ = model(im, text)
        pred.append(bbox_pred[0])
    pred = np.stack(pred, 0)
    evaluator = GroundingEvaluator(gt_dataset=dataset, iou_thresh=iou_thr)
    acc = evaluator(torch.from_numpy(pred))
    print('Acc: {}'.format(acc))

def evalVGPs(model, dataset, sim_thr):
    TruePredCnt = 0
    wrongCases = []
    processed = 0
    for idx in tqdm(range(len(dataset))):
        img = dataset[idx]['image']
        phrases = dataset[idx]['phrases']
        heatmaps = model(img, phrases)
        similarity_score = cosine_similarity(heatmaps[0].reshape(1, -1), heatmaps[1].reshape(1, -1))[0, 0]
        pred = similarity_score > sim_thr
        gt = dataset[idx]['isVGPs'] == 'True'
        TruePredCnt += pred==gt
        processed += 1
        if pred != gt:
            wrongCases.append([idx, phrases, similarity_score, pred, gt])
    # acc = TruePredCnt/len(dataset)
    acc = TruePredCnt / processed
    with open('data/flickr/demo_results.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset_idx', 'phrases', 'sim_score', 'pred', 'gt'])
        writer.writerows(wrongCases)
    print('Acc: {}'.format(acc))


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='vit14')
parser.add_argument('--dataset', type=str, default='flickr_s1')
parser.add_argument('--iou_thr', type=float, default=0.5)
parser.add_argument('--num_samples', type=int,
                    default=0)  # 0 to test all samples
parser_args = parser.parse_args()

if parser_args.model == 'vgp_vit':
    model = VGPViT
    if parser_args.dataset.startswith('flickr'):
        args = {
            'model': 'vit14',
            'alpha': 0.75,
            'aggregation': 'mean',
            'n_segments': list(range(100, 601, 50)),
            'temperature': 0.02,
            'upsample': 2,
            'start_block': 0,
            'compactness': 50,
            'sigma': 0,
        }
    else:
        assert False
elif parser_args.model == 'vit14':
    model = SLICViT
    if parser_args.dataset.startswith('flickr'):
        args = {
            'model': 'vit14',
            'alpha': 0.75,
            'aggregation': 'mean',
            'n_segments': list(range(100, 601, 50)),
            'temperature': 0.02,
            'upsample': 2,
            'start_block': 0,
            'compactness': 50,
            'sigma': 0,
        }
    elif parser_args.dataset.startswith('vg'):
        args = {
            'model': 'vit14',
            'alpha': 0.8,
            'aggregation': 'mean',
            'n_segments': list(range(100, 601, 50)),
            'temperature': 0.01,
            'upsample': 2,
            'start_block': 0,
            'compactness': 50,
            'sigma': 0,
        }
    else:
        assert False
elif parser_args.model == 'vit16':
    model = SLICViT
    if parser_args.dataset.startswith('flickr'):
        args = {
            'model': 'vit16',
            'alpha': 0.8,
            'aggregation': 'mean',
            'n_segments': list(range(100, 601, 50)),
            'temperature': 0.01,
            'upsample': 2,
            'start_block': 0,
            'compactness': 50,
            'sigma': 0,
        }
    elif parser_args.dataset.startswith('vg'):
        args = {
            'model': 'vit16',
            'alpha': 0.85,
            'aggregation': 'mean',
            'n_segments': list(range(100, 601, 50)),
            'temperature': 0.01,
            'upsample': 2,
            'start_block': 0,
            'compactness': 50,
            'sigma': 0,
        }
    else:
        assert False
elif parser_args.model == 'vit32':
    model = SLICViT
    if parser_args.dataset.startswith('flickr'):
        args = {
            'model': 'vit32',
            'alpha': 0.9,
            'aggregation': 'mean',
            'n_segments': list(range(100, 401, 50)),
            'temperature': 0.009,
            'upsample': 2,
            'start_block': 0,
            'compactness': 50,
            'sigma': 0,
        }
    elif parser_args.dataset.startswith('vg'):
        args = {
            'model': 'vit32',
            'alpha': 0.9,
            'aggregation': 'mean',
            'n_segments': list(range(100, 401, 50)),
            'temperature': 0.008,
            'upsample': 2,
            'start_block': 0,
            'compactness': 50,
            'sigma': 0,
        }
    else:
        assert False
elif parser_args.model == 'rn50x4':
    model = ResNetHighRes
    if parser_args.dataset.startswith('flickr'):
        args = {
            'model': 'RN50x4',
            'alpha': 0.7,
            'temperature': 0.03,
        }
    elif parser_args.dataset.startswith('vg'):
        args = {
            'model': 'RN50x4',
            'alpha': 0.7,
            'temperature': 0.01,
        }
    else:
        assert False
elif parser_args.model == 'rn50':
    model = ResNetHighRes
    if parser_args.dataset.startswith('flickr'):
        args = {
            'model': 'RN50',
            'alpha': 0.7,
            'temperature': 0.03,
        }
    elif parser_args.dataset.startswith('vg'):
        args = {
            'model': 'RN50',
            'alpha': 0.8,
            'temperature': 0.01,
        }
    else:
        assert False
elif parser_args.model == 'denseclip':
    model = ResNetHighRes
    if parser_args.dataset.startswith('flickr'):
        args = {
            'model': 'RN50x4',
            'high_res': False,
            'alpha': 0.7,
            'temperature': 0.03,
        }
    elif parser_args.dataset.startswith('vg'):
        args = {
            'model': 'RN50x4',
            'high_res': False,
            'alpha': 0.8,
            'temperature': 0.03,
        }
    else:
        assert False
elif parser_args.model == 'ssbaseline':
    model = SSBaseline
    args = {}
else:
    assert False


# this experiment
if parser_args.dataset == 'flickr_demo':
    dataset = FlickrVGPsDataset(data_type='demo')

# TODO: other splits
elif parser_args.dataset == 'flickr_original':
    dataset = FlickrDataset(data_type='flickr30k/test')
elif parser_args.dataset == 'flickr_s0':
    dataset = FlickrDataset(data_type='flickr30k_c0/test')
elif parser_args.dataset == 'flickr_s1':
    dataset = FlickrDataset(data_type='flickr30k_c1/test')
elif parser_args.dataset == 'flickr_s0_val':
    dataset = FlickrDataset(data_type='flickr30k_c0/val')
elif parser_args.dataset == 'flickr_s1_val':
    dataset = FlickrDataset(data_type='flickr30k_c1/val')
elif parser_args.dataset == 'flickr_other':
    dataset = FlickrDataset(data_type='flickr30k/test', phrase_types=['other'])
elif parser_args.dataset == 'vg_s0':
    dataset = VGDataset(data_type='test_balanced_c2')
elif parser_args.dataset == 'vg_s1':
    dataset = VGDataset(data_type='test_balanced_c3')
elif parser_args.dataset == 'vg_s0_val':
    dataset = VGDataset(data_type='val_balanced')
elif parser_args.dataset == 'vg_s1_val':
    dataset = VGDataset(data_type='val_balanced')
else:
    assert False

model = model(**args).cuda()

if parser_args.num_samples > 0:
    dataset.image_paths = dataset.image_paths[:parser_args.num_samples]

# eval(model, dataset, iou_thr=parser_args.iou_thr)
evalVGPs(model, dataset, sim_thr=0.7)