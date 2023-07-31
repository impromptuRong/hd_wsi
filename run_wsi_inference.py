import os
import sys
import time
import torch
import psutil
import argparse
from PIL import Image
from openslide import open_slide

import configs as CONFIGS
from collections import defaultdict
from utils.utils_image import Slide
from utils.utils_wsi import ObjectIterator, WholeSlideDataset, folder_iterator
from utils.utils_wsi import get_slide_and_ann_file, generate_roi_masks
from utils.utils_wsi import load_cfg, load_hdyolo_model, yolo_inference_iterator
from utils.utils_wsi import export_detections_to_image, export_detections_to_table, wsi_imwrite


# TODO: using multiprocessing.Queue for producer/consumer without IO blocking.
def analyze_one_slide(model, dataset, batch_size=64, n_workers=64, 
                      compute_masks=True, nms_params={}, device=torch.device('cpu'), 
                      export_masks=None, max_mem=None):
    _byte2mb = lambda x: x / 1e6
    max_mem = max_mem or _byte2mb(psutil.virtual_memory().free * 0.8)
    input_size = dataset.model_input_size
    N_patches = len(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        num_workers=n_workers, shuffle=False,
        pin_memory=True,
    )

    model.eval()
    t0 = time.time()
    print(f"Inferencing: ", end="")
    generator = yolo_inference_iterator(
        model, data_loader, 
        input_size=input_size,
        compute_masks=compute_masks, 
        device=device, **nms_params,
    )

    results = defaultdict(list)
    masks, mask_mem, file_index = [], 0, 0
    for o in generator:
        for k, v in o.items():
            if k != 'masks':
                results[k].append(v.cpu())
            else:
                mask_tensor = v.cpu()
                masks.append(mask_tensor)
                mask_mem += _byte2mb(sys.getsizeof(mask_tensor.storage()))
                avail_mem = min(max_mem, _byte2mb(psutil.virtual_memory().free * 0.8))
                # print(f"Track memory usage: {mask_mem}, {mask_mem/max_mem}")
                if export_masks and mask_mem >= avail_mem:
                    file_index += 1
                    filename = f"{export_masks}_{file_index:0{len(str(N_patches))}}"
                    torch.save(torch.cat(masks), filename)
                    while masks:
                        ele = masks.pop()
                        del ele
                    mask_mem = 0
    
    if export_masks and masks:
        if file_index == 0:
            filename = export_masks
        else:
            file_index += 1
            filename = f"{export_masks}_{file_index:0{len(str(N_patches))}}"
        torch.save(torch.cat(masks), filename)
        while masks:
            ele = masks.pop()
            del ele
            mask_mem = 0

    res = {k: torch.cat(v) for k, v in results.items()}
    if masks:
        res['masks'] = torch.cat(masks)
    t1 = time.time()
    print(f"{t1-t0} s")

    return {'cell_stats': res, 'inference_time': t1-t0}


def main(args):
    if args.model in CONFIGS.MODEL_PATHS:
        args.model = CONFIGS.MODEL_PATHS[args.model]
    print("==============================")
    model = load_hdyolo_model(args.model, nms_params=CONFIGS.NMS_PARAMS)
    device = torch.device(args.device)
    print(f"Load model: {args.model} to {args.device} (nms: {model.headers.det.nms_params}")

    meta_info = load_cfg(args.meta_info)
    dataset_configs = {'mpp': CONFIGS.DEFAULT_MPP, **CONFIGS.DATASETS, **meta_info}
    print(f"Dataset configs: {dataset_configs}")

    if os.path.isdir(args.data_path):
        keep_fn = lambda x: os.path.splitext(x)[1] in ['.svs', '.tiff']
        slide_files = list(folder_iterator(args.data_path, keep_fn))
#         slide_files = [os.path.join(args.data_path, _) for _ in os.listdir(args.data_path) 
#                        if not _.startswith('.') and _.endswith('.svs')]
    else:
        rel_path = os.path.basename(args.data_path)
        slide_files = [(0, rel_path, args.data_path)]
    print(f"Inputs: {args.data_path} ({len(slide_files)} files observed). ")
    print(f"Outputs: {args.output_dir}")
    print("==============================")

    for file_idx, rel_path, slide_file in slide_files:
        output_dir = os.path.join(args.output_dir, os.path.dirname(rel_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        slide_id = os.path.splitext(os.path.basename(slide_file))[0]
        res_file = os.path.join(output_dir, f"{slide_id}.pt")
        res_file_masks = os.path.join(output_dir, f"{slide_id}.masks.pt")

        print("==============================")
        if not os.path.exists(res_file):
            t0 = time.time()
            try:
                osr = Slide(*get_slide_and_ann_file(slide_file))
                roi_masks = generate_roi_masks(osr, args.roi)
                dataset = WholeSlideDataset(osr, masks=roi_masks, processor=None, **dataset_configs)
                osr.attach_reader(open_slide(osr.img_file))
                print(dataset.info())
            except Exception as e:
                print(f"Failed to create slide dataset for: {slide_file}.")
                print(e)
                continue
            print(f"Loading slide: {time.time()-t0} s")
            outputs = analyze_one_slide(model, dataset,  
                                        compute_masks=not args.box_only,
                                        batch_size=args.batch_size, 
                                        n_workers=args.num_workers, 
                                        nms_params={}, device=device, 
                                        export_masks=res_file_masks,
                                        max_mem=args.max_memory,
                                       )
            outputs['meta_info'] = meta_info
            outputs['slide_info'] = dataset.slide.info()
            outputs['slide_size'] = dataset.slide_size
            outputs['model'] = args.model
            outputs['rois'] = dataset.masks

            # we save nuclei masks in a separate file to speed up features extraction without mask.
            if 'masks' in outputs['cell_stats']:
                output_masks = outputs['cell_stats']['masks']
                del outputs['cell_stats']['masks']
                torch.save(output_masks, res_file_masks)
                torch.save(outputs, res_file)
                outputs['cell_stats']['masks'] = output_masks
            else:
                torch.save(outputs, res_file)
            osr.detach_reader(close=True)
            print(f"Total time: {time.time()-t0} s")
        else:
            outputs = {}

        if args.save_img or args.save_csv:
            if not outputs:
                outputs = torch.load(res_file)
            if args.box_only:
                param_masks = None
            else:
                if 'masks' in outputs['cell_stats']:
                    param_masks = outputs['cell_stats']['masks']
                elif os.path.exists(res_file_masks):
                    param_masks = torch.load(res_file_masks)
                else:
                    param_masks = res_file_masks
        
        if args.save_img:
            img_file = os.path.join(output_dir, f"{slide_id}.tiff")
            if not os.path.exists(img_file):
                print(f"Exporting result to image: ", end="")
                t0 = time.time()
                
                object_iterator = ObjectIterator(
                    boxes=outputs['cell_stats']['boxes'], 
                    labels=outputs['cell_stats']['labels'], 
                    scores=outputs['cell_stats']['scores'], 
                    masks=param_masks,
                )
                mask = export_detections_to_image(
                    object_iterator, outputs['slide_size'], 
                    labels_color=outputs['meta_info']['labels_color'],
                    save_masks=not args.box_only, border=3, 
                    alpha=1.0 if args.box_only else CONFIGS.MASK_ALPHA,
                )
                # Image.fromarray(mask).save(img_file)
                wsi_imwrite(mask, img_file, outputs['slide_info'], CONFIGS.TIFF_PARAMS,
                            model=outputs['model'],
                           )
                print(f"{time.time()-t0} s")

        if args.save_csv:
            csv_file = os.path.join(output_dir, f"{slide_id}.csv")
            if not os.path.exists(csv_file):
                print(f"Exporting result to csv: ", end="")
                t0 = time.time()

                if args.export_text and 'labels_text' in outputs['meta_info']:
                    labels_text = outputs['meta_info']['labels_text']
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
                    save_masks=(not args.box_only) and args.export_mask,
                )
                df.to_csv(csv_file, index=False)
                print(f"{time.time()-t0} s")
        print("==============================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WSI inference with HD-Yolo.', add_help=True)
    parser.add_argument('--data_path', required=True, type=str, help="input data filename or directory.")
    parser.add_argument('--meta_info', default='meta_info.yaml', type=str, 
                        help="meta info yaml file: contains label texts and colors.")
    parser.add_argument('--model', default='lung', type=str, help="Model path, torch jit model." )
    parser.add_argument('--output_dir', default='slide_results', type=str, help="Output folder.")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], type=str, help='Run on cpu or gpu.')
    parser.add_argument('--roi', default='tissue', type=str, help='ROI region.')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of batch size.')
    parser.add_argument('--num_workers', default=64, type=int, help='Number of workers for data loader.')
    parser.add_argument('--max_memory', default=None, type=int, 
                        help='Maximum MB to store masks in memory. Default is 80%% of free memory.')
    parser.add_argument('--box_only', action='store_true', help="Only save box and ignore mask.")
    parser.add_argument('--save_img', action='store_true', 
                        help="Plot nuclei box/mask into png, don't enable this option for large image.")
    parser.add_argument('--save_csv', action='store_true', 
                        help="Save nuclei information into csv, not necessary.")
    parser.add_argument('--export_text', action='store_true', 
                        help="If save_csv is enabled, whether to convert numeric labels into text.")
    parser.add_argument('--export_mask', action='store_true', 
                        help="If save_csv is enabled, whether to export mask polygons into csv.")

    args = parser.parse_args()
    main(args)
