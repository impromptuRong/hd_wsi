import os
import math
import time
import torch
import numbers
import argparse
from PIL import Image
import torch.nn.functional as F
from torchvision.io import read_image, write_png

import configs as CONFIGS
from utils_wsi import is_image_file, load_cfg, export_detections_to_image, export_detections_to_table
from utils_image import get_pad_width


def analyze_one_patch(img, model, dataset_configs, mpp=None, 
                      compute_masks=True, device=torch.device('cpu')):
    h_ori, w_ori = img.shape[1:]
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

    if device.type == 'cpu':  # half precision only supported on CUDA
        model.float()
    model.eval()
    model.to(device)

    t0 = time.time()
    with torch.no_grad():
        inputs = inputs.to(device, next(model.parameters()).dtype, non_blocking=True)
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


def main(args):
    if args.model in CONFIGS.MODEL_PATHS:
        args.model = CONFIGS.MODEL_PATHS[args.model]
    model = torch.jit.load(args.model)
    device = torch.device(args.device)

    meta_info = load_cfg(args.meta_info)
    dataset_configs = {'mpp': CONFIGS.DEFAULT_MPP, **CONFIGS.DATASETS, **meta_info}

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if os.path.isdir(args.data_path):
        patch_files = [os.path.join(args.data_path, _) for _ in os.listdir(args.data_path) 
                       if is_image_file(_)]
    else:
        patch_files = [args.data_path]

    for patch_path in patch_files:
        print("==============================")
        print(patch_path)
        image_id, ext = os.path.splitext(os.path.basename(patch_path))
        # run inference
        img = read_image(patch_path).type(torch.float32) / 255
        outputs = analyze_one_patch(
            img, model, dataset_configs, mpp=args.mpp, 
            compute_masks=not args.box_only, device=device,
        )
        print(f"Inference time: {outputs['inference_time']} s")
        res_file = os.path.join(args.output_dir, f"{image_id}_pred.pt")
        torch.save(outputs, res_file)

        # save image
        mask_img = export_detections_to_image(
            outputs['cell_stats'], (img.shape[1], img.shape[2]), 
            labels_color=dataset_configs['labels_color'],
            save_masks=not args.box_only, border=3,
            alpha=1.0 if args.box_only else CONFIGS.MASK_ALPHA,
        )
        img_file = os.path.join(args.output_dir, f"{image_id}_pred{ext}")
        Image.fromarray(mask_img).save(img_file)
        # write_png((img_mask*255).type(torch.uint8), img_file)

        # save to csv
        if args.export_text and 'labels_text' in dataset_configs:
            labels_text = dataset_configs['labels_text']
        else:
            labels_text = None
        df = export_detections_to_table(
            outputs['cell_stats'], 
            labels_text=labels_text,
            save_masks=not args.box_only,
        )
        csv_file = os.path.join(args.output_dir, f"{image_id}_pred.csv")
        df.to_csv(csv_file, index=False)
        print("==============================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Patch inference with HD-Yolo.', add_help=True)
    parser.add_argument('--data_path', required=True, type=str, help="Input data filename or directory.")
    parser.add_argument('--meta_info', default='meta_info.yaml', type=str, 
                        help="A yaml file contains: label texts and colors.")
    parser.add_argument('--model', default='brca', type=str, help="Model path, torch jit model." )
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
