import os
import cv2
import math
import time
import torch
import numbers
import argparse
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import skimage.io

import configs as CONFIGS
from utils.utils_wsi import ObjectIterator, folder_iterator
from utils.utils_wsi import load_cfg, load_hdyolo_model, is_image_file
from utils.utils_wsi import export_detections_to_image, export_detections_to_table
from utils.utils_image import get_pad_width, rgba2rgb


def analyze_one_patch(img, model, dataset_configs, mpp=None, compute_masks=True,):
    h_ori, w_ori = img.shape[1:]
    model_par0 = next(model.parameters())
    img = img.to(model_par0.device, model_par0.dtype, non_blocking=True)

    ## rescale
    if mpp is not None and mpp != dataset_configs['mpp']:
        scale_factor = dataset_configs['mpp'] / mpp
        img_rescale = F.interpolate(img[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0] 
    else:
        scale_factor = 1.0
        img_rescale = img
    h_rescale, w_rescale = img_rescale.shape[1:]

    ## pad to 64
    if h_rescale % 64 != 0 or w_rescale % 64 != 0:
        input_h, input_w = math.ceil(h_rescale / 64) * 64, math.ceil(w_rescale / 64) * 64
        pad_width = get_pad_width((h_rescale, w_rescale), (input_h, input_w), pos='center', stride=1)
        inputs = F.pad(img_rescale[None], [pad_width[1][0], pad_width[1][1], pad_width[0][0], pad_width[0][1]], 
                       mode='constant', value=0.)
    else:
        pad_width = [(0, 0), (0, 0)]
        inputs = img_rescale[None]

    t0 = time.time()
    with torch.no_grad():
        outputs = model(inputs, compute_masks=compute_masks)[1]
        res = outputs[0]['det']

    ## unpad and scale back to original coords
    res['boxes'] -= res['boxes'].new([pad_width[1][0], pad_width[0][0], pad_width[1][0], pad_width[0][0]])
    res['boxes'] /= scale_factor
    res['labels'] = res['labels'].to(torch.int32)
    res['boxes'] = res['boxes'].to(torch.float32)
    res = {k: v.cpu().detach() for k, v in res.items()}
    t1 = time.time()

    return {'cell_stats': res, 'inference_time': t1-t0}


def overlay_masks_on_image(image, mask):
    msk = Image.fromarray(mask)
    blended = Image.fromarray(image)
    blended.paste(msk, mask=msk.split()[-1])
    
    return blended


def main(args):
    if args.model in CONFIGS.MODEL_PATHS:
        args.model = CONFIGS.MODEL_PATHS[args.model]
    print("==============================")
    model = load_hdyolo_model(args.model, nms_params=CONFIGS.NMS_PARAMS)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"Cuda is not available, use cpu instead.")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    if device.type == 'cpu':  # half precision only supported on CUDA
        model.float()
    model.eval()
    model.to(device)
    print(f"Load model: {args.model} to {args.device} (nms: {model.headers.det.nms_params}")

    meta_info = load_cfg(args.meta_info)
    dataset_configs = {'mpp': CONFIGS.DEFAULT_MPP, **CONFIGS.DATASETS, **meta_info}
    print(f"Dataset configs: {dataset_configs}")

    if os.path.isdir(args.data_path):
        keep_fn = lambda x: is_image_file(x)
        patch_files = list(folder_iterator(args.data_path, keep_fn))
    else:
        rel_path = os.path.basename(args.data_path)
        patch_files = [(0, rel_path, args.data_path)]
    print(f"Inputs: {args.data_path} ({len(patch_files)} files observed). ")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Outputs: {args.output_dir}")
    print("==============================")

    for file_idx, rel_path, patch_path in patch_files:
        print("==============================")
        print(patch_path)
        output_dir = os.path.join(args.output_dir, os.path.dirname(rel_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_id, ext = os.path.splitext(os.path.basename(patch_path))
        # run inference
        # img = read_image(patch_path).type(torch.float32) / 255
        raw_img = rgba2rgb(skimage.io.imread(patch_path))
        # raw_img = cv2.resize(raw_img, (500, 500), interpolation=cv2.INTER_LINEAR)
        img = ToTensor()(raw_img)
        outputs = analyze_one_patch(
            img, model, dataset_configs, mpp=args.mpp, 
            compute_masks=not args.box_only,
        )
        print(f"Inference time: {outputs['inference_time']} s")
        res_file = os.path.join(output_dir, f"{image_id}_pred.pt")
        torch.save(outputs, res_file)

        if 'masks' in outputs['cell_stats']:
            param_masks = outputs['cell_stats']['masks']
        else:
            param_masks = None

        # save image
        object_iterator = ObjectIterator(
            boxes=outputs['cell_stats']['boxes'], 
            labels=outputs['cell_stats']['labels'], 
            scores=outputs['cell_stats']['scores'], 
            masks=param_masks,
        )
        mask_img = export_detections_to_image(
            object_iterator, (img.shape[1], img.shape[2]), 
            labels_color=dataset_configs['labels_color'],
            save_masks=not args.box_only, border=3,
            alpha=1.0 if args.box_only else CONFIGS.MASK_ALPHA,
        )
        export_img = overlay_masks_on_image(raw_img, mask_img)
        img_file = os.path.join(output_dir, f"{image_id}_pred{ext}")
        export_img.save(img_file)
        # Image.fromarray(mask_img).save(img_file)
        # write_png((img_mask*255).type(torch.uint8), img_file)

        # save to csv
        if args.export_text and 'labels_text' in dataset_configs:
            labels_text = dataset_configs['labels_text']
        else:
            labels_text = None
        object_iterator = ObjectIterator(
            boxes=outputs['cell_stats']['boxes'], 
            labels=outputs['cell_stats']['labels'], 
            scores=outputs['cell_stats']['scores'], 
            masks=param_masks,
        )
        df = export_detections_to_table(
            object_iterator, 
            labels_text=labels_text,
            save_masks=not args.box_only,
        )
        csv_file = os.path.join(output_dir, f"{image_id}_pred.csv")
        df.to_csv(csv_file, index=False)
        print("==============================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Patch inference with HD-Yolo.', add_help=True)
    parser.add_argument('--data_path', required=True, type=str, help="Input data filename or directory.")
    parser.add_argument('--meta_info', default='meta_info.yaml', type=str, 
                        help="A yaml file contains: label texts and colors.")
    parser.add_argument('--model', default='lung', type=str, help="Model path, torch jit model." )
    parser.add_argument('--output_dir', default='patch_results', type=str, help="Output folder.")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], type=str, help='Run on cpu or gpu.')
    parser.add_argument('--mpp', default=None, type=float, help='Input data mpp.')
    # parser.add_argument('--batch_size', default=64, type=int, help='Number of batch size.')
    # parser.add_argument('--num_workers', default=64, type=int, help='Number of workers for data loader.')
    parser.add_argument('--box_only', action='store_true', help="Only save box and ignore mask.")
    parser.add_argument('--export_text', action='store_true', 
                        help="If save_csv is enabled, whether to convert numeric labels into text.")

    args = parser.parse_args()
    main(args)
