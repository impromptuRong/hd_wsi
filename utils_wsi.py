import os
import sys
import math
import time
import random
import pickle
import numbers
import skimage

import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from torchvision.transforms import ToTensor
from torchvision.models.detection.roi_heads import paste_masks_in_image

import cv2
import numpy as np
import pandas as pd
import multiprocessing as mp

import matplotlib
import tifffile
from matplotlib import pyplot as plt
from utils_image import Slide, img_as, rgba2rgb, pad, Mask, overlay_detections, image_stats
from openslide import open_slide


TO_REMOVE = 1


def load_cfg(cfg):
    if isinstance(cfg, dict):
        yaml = cfg
    else:
        import yaml
        with open(cfg, encoding='ascii', errors='ignore') as f:
            yaml = yaml.safe_load(f)  # model dict

    return yaml


def collate_fn(batch):
    return tuple(zip(*batch))


def is_image_file(x):
    ext = os.path.splitext(x)[1].lower()
    return not x.startswith('.') and ext in ['.png', '.jpeg', '.jpg', '.tif', '.tiff']


def get_slide_and_ann_file(svs_file, ann_file=None):
    folder_name, file_name = os.path.split(svs_file)
    slide_id, ext = os.path.splitext(file_name)
    if ann_file is True:
        ann_file = os.path.join(folder_name, slide_id + '.xml')
    if not (isinstance(ann_file, str) and os.path.exists(ann_file)):
        ann_file = None

    return slide_id, svs_file, ann_file


def processor(patch, info, **kwargs):
    scale = kwargs.get('scale', 1.0)
    if scale != 1.0:
        h_new, w_new = int(round(patch.shape[0] * scale)), int(round(patch.shape[1] * scale))
        patch = cv2.resize(patch, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        # patch = skimage.transform.rescale(patch, scale, order=3, multichannel=True)
        info = {
            'roi_slide': info['roi_slide'] * scale, 
            'roi_patch': info['roi_patch'] * scale,
        }

    return patch, info


class WholeSlideDataset(torch.utils.data.Dataset):
    def __init__(self, svs_file, patch_size=512, padding=64, mpp=0.25, page=0, 
                 ann_file=None, masks=None, processor=None, **kwargs):
        self.slide_id, self.svs_file, self.ann_file = get_slide_and_ann_file(svs_file, ann_file)

        self.slide = Slide(self.svs_file, self.ann_file)
        self.slide_handle = open_slide(self.svs_file)
        self.page = page
        self.slide_size = [self.slide.level_dims[page][1], self.slide.level_dims[page][0]]
        assert padding % (patch_size/64) == 0, f"Padding {padding} should be divisible by {patch_size/64}."
        self.patch_size = patch_size
        self.padding = padding
        # self.mpp = mpp
        self.model_input_size = self.patch_size + self.padding * 2

        self.processor = processor
        self.kwargs = kwargs

        # self.kwargs['width'], self.kwargs['height'] = self.slide.level_dims[0]
        # self.kwargs['magnitude'] = self.slide.magnitude
        if self.slide.mpp != mpp:  # we use mpp instead of magnitude
            self.window_size = int(round(self.patch_size * mpp/self.slide.mpp / 64) * 64)
            self.scale = self.patch_size / self.window_size
            self.window_padding = int(self.padding / self.scale)
        else:
            self.scale = 1.0
            self.window_size = self.patch_size
            self.window_padding = self.padding

        # st = time.time()
        if isinstance(masks, str):
            if masks == 'tissue':
                self.masks = self.slide.roughly_extract_tissue_region((512, 512), bg=255)
            elif masks == 'all':
                self.masks = None
            elif masks == 'xml':
                self.masks = self.slide.get_annotations(pattern='.*')
            else:
                self.masks = self.slide.get_annotations(pattern=masks)
        elif callable(masks):
            level_idx = self.slide.get_resize_level((1024, 1024))
            img = self.slide.get_page(level=level_idx)
            self.masks = masks(img)
        else:
            self.masks = masks
        # print(f"generate masks: {time.time()-st}")

        self.patches, self.polygons, self.poly_indices = self.slide.whole_slide_scanner(
            self.window_size, self.page, masks=self.masks, coverage_threshold=0.,
        )
        pars = self.slide.pad_roi(self.patches, self.window_size, self.page, padding=self.window_padding)
        pars += (self.poly_indices,)

        self.images = []
        for idx, (coord, roi_slide, roi_patch, pad_width, poly_id) in enumerate(zip(*pars), 1):
            image_info = {
                'image_id': f'{self.slide_id}_{idx:05d}', 
                'data': {
                    'coord': coord, 
                    'roi_slide': roi_slide,
                    'roi_patch': roi_patch,
                    'pad_width': pad_width, 
                    'poly_id': poly_id,
                }, 
                'kwargs': {},
            }
            self.images.append(image_info)

    def load_patch(self, idx):
        info = self.images[idx]['data']
        patch = np.array(self.slide.get_patch(self.slide_handle, info['coord'], 0))
        patch = img_as('float32')(rgba2rgb(patch))
        pad_width = [(info['pad_width'][0], info['pad_width'][1]),
                     (info['pad_width'][2], info['pad_width'][3])]
        patch = pad(patch, pad_width=pad_width, mode='constant', cval=0.0)

        return patch

    def __getitem__(self, idx):
        patch = self.load_patch(idx)
        info = self.images[idx]['data']
        roi_slide = torch.from_numpy(info['roi_slide'].astype(np.int32))
        roi_patch = torch.from_numpy(info['roi_patch'].astype(np.int32))
        ## don't resize, keep it consistent with original slide scale.
        # patch = cv2.resize(patch, (self.model_input_size, self.model_input_size), interpolation=cv2.INTER_LINEAR)

        if self.processor is not None:
            kwargs = {**self.kwargs, **self.images[idx]['kwargs']}
            patch = self.processor(patch **kwargs)

        return ToTensor()(patch), [roi_slide, roi_patch]

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"WholeSlideDataset: {len(self)} patches."

    def __str__(self):
        return self.info()

    def info(self):
        slide_info = f"{self.slide_id}: {self.slide.magnitude}x"
        mpp_info = f"mpp: {self.slide.mpp}->{self.slide.mpp/self.scale}"
        inference_size = [math.ceil(self.slide_size[0] * self.scale), math.ceil(self.slide_size[1] * self.scale)]
        size_info = f"size: {self.slide_size}->{inference_size}"
        patch_info = f"patch: {self.window_size}({self.window_padding})->{self.patch_size}({self.padding})"
        roi_area = self.masks.sum()/self.masks.size if self.masks is not None else 1.0
        mask_info = f"roi: {roi_area*100:.2f}% ({len(self)} patches)"

        return f"{slide_info}, {mpp_info}, {size_info}, {patch_info}, {mask_info}"

    def export(self, output_folder, save_thumbnail=False, save_mask=False):
        data_file_name = os.path.join(output_folder, f'{self.slide_id}.pkl')
        with open(data_file_name, 'wb') as f:
            info = {
                'patch_size': self.patch_size, 'padding': self.padding, 
                'patches': self.images, 'kwargs': self.kwargs,
            }
            pickle.dump(info, f, protocol=(pickle.HIGHEST_PROTOCOL))

        if save_thumbnail:
            image_file_name = os.path.join(output_folder, f'{self.slide_id}.png')
            self.slide.thumbnail().save(image_file_name)
            # skimage.io.imsave(image_file_name, np.array(self.slide.thumbnail()))

        if save_mask and self.masks is not None:
            mask_file_name = os.path.join(output_folder, f'{self.slide_id}_mask.png')
            if isinstance(self.masks, np.ndarray):  # mask image
                mask_img = self.masks * 1.0
            elif isinstance(masks, list):  # polygon
                w_m, h_m = self.slide.level_dims[0]
                mask_img = self.slide.polygons2mask(self.masks, (w_m/32, h_m/32), scale=1./32)
            cv2.imwrite(mask_file_name, self.masks * 1.0)
            # skimage.io.imsave(mask_file_name, self.masks * 1.0)

    def load(self, folder):
        data_file_name = os.path.join(folder, f'{self.slide_id}.pkl')
        mask_file_name = os.path.join(folder, f'{self.slide_id}_mask.png')

        if os.path.exists(data_file_name):
            with open(data_file_name, 'rb') as f:
                self.images = pickle.load(f)

        if os.path.exists(mask_file_name):
            self.masks = img_as('bool')(skimage.io.imread(mask_file_name))

    def display(self, indices=None, results=None):
        if indices is None:
            indices = range(len(self))
        elif isinstance(indices, numbers.Number):
            indices = np.random.choice(len(self), indices)

        labels_color = self.kwargs['labels_color']
        labels_text = self.kwargs['labels_text']
        for idx in indices:
            patch_id = self.images[idx]['image_id']
            patch, patch_info = self[idx]
            patch = patch.permute(1, 2, 0).numpy()
            h, w = patch.shape[0], patch.shape[1]

            print('===================')
            print(patch_id, patch_info)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(patch)
            if results is not None:
                output = results[idx]
                o_boxes = output['boxes'].numpy()
                o_labels = output['labels'].numpy()
                if 'masks' in output and len(o_boxes) > 0:
                    o_masks = paste_masks_in_image(output['masks'], output['boxes'], (h, w), padding=1).squeeze(1).numpy()
                else:
                    o_masks = None
                axes[1].imshow(patch)
                overlay_detections(
                    axes[1], bboxes=o_boxes, labels=o_labels, masks=o_masks, scores=None, 
                    labels_color=labels_color, labels_text=labels_text,
                    show_bboxes=True, show_texts=False, show_masks=True, show_scores=False
                )
            else:
                plt.show()


def yolov5_inference(model, data_loader, input_size=640, compute_masks=True, device=torch.device('cuda'), **kwargs):
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
    model.eval()
    model.to(device)

    results = defaultdict(list)
    with torch.no_grad():
        for images, patch_infos in data_loader:
            images = images.to(device, next(model.parameters()).dtype, non_blocking=True)
            h_ori, w_ori = images.shape[-2], images.shape[-1]
            if h_ori != h or w_ori != w:
                inputs = F.interpolate(images, size=(h, w), mode='bilinear', align_corners=False)
            else:
                inputs = images
            _, preds = model(inputs, compute_masks=compute_masks)  # MaskRCNN and Yolo always return loss, preds

            for pred, info in zip(preds, zip(*patch_infos)):
                if 'det' in pred:
                    o = pred['det']  # yolo_mask model, multi-task
                else:
                    o = pred  # maskrcnn model

                if score_threshold > 0.:
                    keep = o['scores'] >= score_threshold
                    o = {k: v[keep] for k, v in o.items()}

                if iou_threshold < 1.:
                    keep = torchvision.ops.nms(o['boxes'], o['scores'], iou_threshold=iou_threshold)
                    o = {k: v[keep] for k, v in o.items()}

                # trim border objects, map to original coords
                o['boxes'][:, [0, 2]] *= w_ori/w  # rescale to image size
                o['boxes'][:, [1, 3]] *= h_ori/h
                roi_slide, roi_patch = info
                y0_s, x0_s, w_s, h_s = roi_slide  # torch.int32
                y0_p, x0_p, w_p, h_p = roi_patch  # torch.int32

                if len(o['boxes']):
                    y_c, x_c = o['boxes'][:,[0,2]].mean(1), o['boxes'][:,[1,3]].mean(1)
                    keep = (y_c > y0_p) & (y_c < y0_p + w_p) & (x_c > x0_p) & (x_c < x0_p + h_p)
                    o = {k: v[keep] for k, v in o.items()}
                    o['labels'] = o['labels'].to(torch.int32)
                    # max number of float16 is 65536, half will lead to inf for large image.
                    o['boxes'] = o['boxes'].to(torch.float32)
                    o['boxes'][:, [0, 2]] += y0_s - y0_p
                    o['boxes'][:, [1, 3]] += x0_s - x0_p

                    # results.append({k: v.cpu() for k, v in o.items()})
                    for k, v in o.items():
                        results[k].append(v.cpu())

    return {k: torch.cat(v) for k, v in results.items()}


def export_detections_to_table(res, converter=None, labels_text=None, save_masks=True):
    boxes, labels, scores = res['boxes'], res['labels'], res['scores']
    if 'masks' in res and save_masks:
        masks = res['masks']
        columns = ['x0', 'y0', 'x1', 'y1', 'score', 'label', 'poly_x', 'poly_y',]
    else:
        masks = [None] * len(boxes)
        columns = ['x0', 'y0', 'x1', 'y1', 'score', 'label',]

    df = []
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        box = box.round().to(torch.int32)
        w = max((box[2] - box[0] + TO_REMOVE).item(), 1)
        h = max((box[3] - box[1] + TO_REMOVE).item(), 1)
        label = label.item() if labels_text is None else labels_text.get(label.item(), f'cls_{label.item()}')
        entry = box.tolist() + [round(score.item(), 4), label]

        if mask is not None:
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


def export_detections_to_image(res, img_size, labels_color, save_masks=True, border=3, alpha=1.0):
    h_s, w_s = img_size
    boxes, labels = res['boxes'], res['labels']
    masks = res['masks'] if ('masks' in res and save_masks) else [None] * len(boxes)
    max_val = labels.max().item()

    color_tensor = torch.zeros((max_val + 1 + 1, 4), dtype=torch.uint8)
    for k, v in labels_color.items():
        if k > 0 and k <= max_val:
            color_tensor[k] = torch.tensor(matplotlib.colors.to_rgba(v)) * 255
        else:
            color_tensor[-1] = torch.tensor(matplotlib.colors.to_rgba(v)) * 255  # outlier class
    color_tensor[..., -1] = color_tensor[..., -1] * alpha  # apply extra transparency

    img_label = torch.zeros((h_s, w_s), dtype=torch.int)
    for box, label, mask in zip(boxes, labels, masks):
        box = box.round().to(torch.int32)
        w = max((box[2] - box[0] + TO_REMOVE).item(), 1)
        h = max((box[3] - box[1] + TO_REMOVE).item(), 1)
        x_0 = max(box[0], 0)
        x_1 = min(box[2] + 1, w_s)
        y_0 = max(box[1], 0)
        y_1 = min(box[3] + 1, h_s)

        if label > 0:  # we ignore unclassified
            if mask is not None:  # draw mask
                # mask = F.interpolate(mask[None].float(), size=(h, w), mode="bilinear", align_corners=False)[0][0]
                mask = cv2.resize(mask[0].float().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5) * (label.item() if label > 0 else max_val + 1)
                mask = torch.from_numpy(mask[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])])
            else:  # draw rectangle
                mask = torch.ones((y_1-y_0, x_1-x_0)) * label
                mask[border:-border, border:-border] = 0
            img_label[y_0:y_1, x_0:x_1] = torch.maximum(img_label[y_0:y_1, x_0:x_1], mask)
            # img_label[y_0:y_1, x_0:x_1] = mask # box will overwrite seg with 0

    return F.embedding(img_label, color_tensor).numpy()


def wsi_imwrite(image, filename, slide_info, tiff_params, **kwargs):
    w0, h0 = image.shape[1], image.shape[0]
    tile_w, tile_h = tiff_params['tile']
    mpp = slide_info['mpp']
    # software='Aperio Image Library v11.1.9'

    with tifffile.TiffWriter(filename, bigtiff=True) as tif:
        descp = f"HD-Yolo\n{w0}x{h0} ({tile_w}x{tile_h}) RGBA|MPP = {mpp}"
        for k, v in kwargs.items():
            descp += f'|{k} = {v}'
        # resolution=(mpp * 1e-4, mpp * 1e-4, 'CENTIMETER')
        tif.save(image, metadata=slide_info, description=descp, **tiff_params,)

        for w, h in sorted(slide_info['level_dims'][1:], key=lambda x: x[0], reverse=True):
            image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            descp = f"{w0}x{h0} ({tile_w}x{tile_h}) -> {w}x{h} RGBA"
            # tile = (page.tilewidth, page.tilelength) if page.tilewidth > 0 and page.tilelength > 0 else None
            # resolution=(mpp * 1e-4 * w0/w, mpp * 1e-4 * h0/h, 'CENTIMETER')
            # 'subfiletype': 1 if level else 0,
            tif.save(image, description=descp, **tiff_params,)
