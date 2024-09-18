import os
import sys
import math
import time
import random
import pickle
import numbers
import skimage
import datetime

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from collections import defaultdict
from torchvision.models.detection.roi_heads import paste_masks_in_image

import cv2
import numpy as np
import pandas as pd
import multiprocessing as mp

import matplotlib
import tifffile
from collections import deque
from matplotlib import pyplot as plt
from .utils_image import Slide, get_dzi, img_as, pad, Mask, overlay_detections

from itertools import compress
from ultralytics import YOLO


TO_REMOVE = 1


def load_cfg(cfg):
    if isinstance(cfg, dict):
        yaml = cfg
    else:
        import yaml
        with open(cfg, encoding='ascii', errors='ignore') as f:
            yaml = yaml.safe_load(f)  # model dict

    return yaml


def load_yolov8_model(model, device='cpu', nms_params={}):
    if isinstance(model, str):
        model = YOLO(model)
        model.fuse()
        model.nms_params = {}
        for k, v in nms_params.items():
            if 'conf_thres' in nms_params:
                model.nms_params['conf'] = nms_params['conf_thres']
            elif 'iou_thres' in nms_params:
                model.nms_params['iou'] = nms_params['iou_thres']
            else:
                model.nms_params[k] = v
    
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == 'cpu':  # half precision only supported on CUDA
        model.float()
    else:
        model.half()
    model.to(device)
    
    return model


def batch_inference(model, images, patch_infos, input_size, compute_masks=True, 
                    score_threshold=0., iou_threshold=1.):
    """ Run a bath inference. 
        Generally, model already has nms integrated, so no need to change 
        default score_threshold, iou_threshold. 
    """
    h, w = input_size
    h_ori, w_ori = images.shape[-2], images.shape[-1]
    if h_ori != h or w_ori != w:
        inputs = F.interpolate(images, size=(h, w), mode='bilinear', align_corners=False)
    else:
        inputs = images

    res = []
    outputs = model(inputs, verbose=False, **model.nms_params)
    for r, info in zip(outputs, patch_infos):
        r = r.cpu()
        o = pred = {
            'boxes': r.boxes.xyxy.clone(), 
            'labels': r.boxes.cls.clone() + 1, 
            'scores': r.boxes.conf.clone(), 
            'masks': r.masks.xy if r.masks else [], 
        }

        if score_threshold > 0.:
            keep = o['scores'] >= score_threshold
            o = {k: list(compress(v, keep)) if isinstance(v, list) else v[keep] for k, v in o.items()}

        if iou_threshold < 1.:
            keep = torchvision.ops.nms(o['boxes'], o['scores'], iou_threshold=iou_threshold)
            o = {k: [v[keep] for idx in keep] if isinstance(v, list) else v[keep] for k, v in o.items()}

        if len(o['boxes']):
            # trim border objects, map to original coords
            o['boxes'][:, [0, 2]] *= w_ori/w  # rescale to image size
            o['boxes'][:, [1, 3]] *= h_ori/h
            
            if 'masks' in o:
                o['masks'] = [[m * np.array([w_ori/w, h_ori/h]) - box[:2].numpy()]
                              for m, box in zip(o['masks'], o['boxes'])]

            x0_s, y0_s, w_p, h_p, x0_p, y0_p = info
            x_c, y_c = o['boxes'][:,[0,2]].mean(1), o['boxes'][:,[1,3]].mean(1)
            keep = (x_c > x0_p) & (x_c < x0_p + w_p) & (y_c > y0_p) & (y_c < y0_p + h_p)
            o = {k: list(compress(v, keep)) if isinstance(v, list) else v[keep] for k, v in o.items()}
            o['labels'] = o['labels'].to(torch.int32)
            # max number of float16 is 65536, half will lead to inf for large image.
            o['boxes'] = o['boxes'].to(torch.float32)
            o['boxes'][:, [0, 2]] += x0_s - x0_p
            o['boxes'][:, [1, 3]] += y0_s - y0_p

        res.append(o)

    return res


def yolo_inference_iterator(model, data_loader, input_size=640, compute_masks=True, device=torch.device('cuda'), **kwargs):
    """ Inference on a whole slide data loader with given model.
        Provide score_threshold and iou_threshold if they are different from default.
    """
    score_threshold = kwargs.get('score_threshold', 0.)
    iou_threshold = kwargs.get('iou_threshold', 1.)
    if isinstance(input_size, numbers.Number):
        h, w = input_size, input_size
    else:
        h, w = input_size

    if device.type == 'cpu':  # half precision only supported on CUDA
        model.float()
    else:
        model.half()
    model.to(device)
    model_dtype = next(model.parameters()).dtype

    results = defaultdict(list)
    with torch.no_grad():
        for images, patch_infos in data_loader:
            images = images.to(device, model_dtype, non_blocking=True)
            r = batch_inference(model, images, patch_infos, 
                                input_size=(h, w), compute_masks=compute_masks, 
                                score_threshold=score_threshold, 
                                iou_threshold=iou_threshold)
            for o in r:
                yield o


def yolov8_inference(model, data_loader, input_size=640, compute_masks=True, device=torch.device('cuda'), **kwargs):
    """ Inference on a whole slide data loader with given model.
        Provide score_threshold and iou_threshold if they are different from default.
    """
    generator = yolo_inference_iterator(
        model, data_loader, input_size=input_size, 
        compute_masks=compute_masks, device=device, **kwargs,
    )
    
    results = defaultdict(list)
    for o in generator:
        for k, v in o.items():
            if k == 'masks':
                results['masks'].extend(v)
            else:
                results[k].append(v.cpu())

    return {k: torch.cat(v) if k != 'masks' else v 
            for k, v in results.items()}


# TODO: using multiprocessing.Queue for producer/consumer without IO blocking.
def analyze_one_slide(model, dataset, batch_size=64, n_workers=64, 
                      compute_masks=True, nms_params={}, device=torch.device('cpu'), 
                      export_masks=None, max_mem=None):
    input_size = dataset.model_input_size
    N_patches = len(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        num_workers=n_workers, shuffle=False,
        pin_memory=True,
    )

    t0 = time.time()
    print(f"Inferencing: ", end="")
    generator = yolo_inference_iterator(
        model, data_loader, 
        input_size=input_size,
        compute_masks=compute_masks, 
        device=device, **nms_params,
    )

    results = defaultdict(list)
    masks = []
    for o in generator:
        for k, v in o.items():
            if k != 'masks':
                results[k].append(v.cpu())
            else:
                masks.extend(v)

    res = {k: torch.cat(v) for k, v in results.items()}
    if masks:
        res['masks'] = masks
    t1 = time.time()
    print(f"{t1-t0} s")

    return {'cell_stats': res, 'inference_time': t1-t0}


def export_detections_to_table(object_iterator, converter=None, labels_text=None, save_masks=True):
    # boxes, labels, scores = res['boxes'], res['labels'], res['scores']
    if save_masks:
        columns = ['x0', 'y0', 'x1', 'y1', 'score', 'label', 'poly_x', 'poly_y',]
    else:
        columns = ['x0', 'y0', 'x1', 'y1', 'score', 'label',]

    df = []
    for obj in object_iterator:
        box, label, score, mask = obj['box'], obj['label'], obj['score'], obj['mask']
        box = box.round().to(torch.int32)
        w = max((box[2] - box[0] + TO_REMOVE).item(), 1)
        h = max((box[3] - box[1] + TO_REMOVE).item(), 1)
        label = label.item() if labels_text is None else labels_text.get(label.item(), f'cls_{label.item()}')
        entry = box.tolist() + [round(score.item(), 4), label]

        if save_masks and mask is not None:
            if isinstance(mask, list):  # we got a polygon
                mask = Mask(mask, size=[h, w], mode='poly').poly()
            else:
                # mask = F.interpolate(mask[None].float(), size=(h, w), mode="bilinear", align_corners=False)[0][0]
                mask = cv2.resize(mask[0].float().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
                mask = Mask(mask > 0.5, size=[h, w], mode='mask').poly()
            if mask.m:
                poly = (mask.m[0] + [box[0], box[1]]).T
                poly_x = ','.join([f'{_:.2f}' for _ in poly[0]])
                poly_y = ','.join([f'{_:.2f}' for _ in poly[1]])
            else:
                poly_x = poly_y = ''
            entry += [poly_x, poly_y]

        df.append(entry)

    return pd.DataFrame(df, columns=columns)

