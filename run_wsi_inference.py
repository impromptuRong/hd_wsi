import os
import time
import torch
import argparse
from PIL import Image
from utils_wsi import DATASET_CONFIGS, WholeSlideDataset
from utils_wsi import yolov5_inference, load_cfg, export_detections_to_image, export_detections_to_table


DATA_PATH = os.path.join(
    "/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/pathology_image_data",
    "TCGA_BRCA/slides/TCGA-Breast Cancer",
)
MODEL_PATH = "./selected_models/benchmark_nucls_paper/fold3_epoch201.float16.torchscript.pt"

# nms_params = {
#     'conf_thres': 0.15, # score_threshold, discards boxes with score < score_threshold
#     'iou_thres': 0.45, # iou_threshold, discards all overlapping boxes with IoU > iou_threshold
#     'classes': None, 
#     'agnostic': True, # False
#     'multi_label': False, 
#     'labels': (), 
#     'max_det': 300, 
# }

ROI_NAMES = {
    'tissue': True,  # use tissue region as roi
    'xml': '.*',  # use all annotations in slide_id.xml 
}

def analyze_one_slide(model, dataset, 
                      batch_size=64, n_workers=64, 
                      compute_masks=True, nms_params={}, 
                      device=torch.device('cpu')):
    input_size = dataset.model_input_size
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        num_workers=n_workers, shuffle=False,
    )

    model.eval()
    t0 = time.time()
    res = yolov5_inference(
        model, data_loader, 
        input_size=input_size,
        compute_masks=compute_masks, 
        device=device, **nms_params,
    )
    t1 = time.time()
    print(f"Inference time: {t1-t0} s")

    return {'cell_stats': res, 'inference_time': t1-t0}


def main(args):
    model = torch.jit.load(args.model_path)
    device = torch.device(args.device)
    
    meta_info = load_cfg(args.meta_info)
    dataset_configs = {**DATASET_CONFIGS, **meta_info}
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if os.path.isdir(args.data_path):
        slide_files = [os.path.join(args.data_path, _) for _ in os.listdir(args.data_path) 
                       if not _.startswith('.') and _.endswith('.svs')]
    else:
        slide_files = [args.data_path]
    
    for slide_file in slide_files:
        svs_file = os.path.join(slide_file)
        slide_id = os.path.splitext(os.path.basename(slide_file))[0]
        res_file = os.path.join(args.output_dir, f"{slide_id}.pt")
        res_file_masks = os.path.join(args.output_dir, f"{slide_id}.masks.pt")
        
        outputs = {}
        print("==============================")
        if not os.path.exists(res_file):
            t0 = time.time()
            dataset = WholeSlideDataset(svs_file, masks=args.roi, processor=None, **dataset_configs)
            print(dataset.info())
            outputs = analyze_one_slide(model, dataset,  
                                        compute_masks=not args.box_only,
                                        batch_size=args.batch_size, 
                                        n_workers=args.num_workers, 
                                        nms_params={}, device=device)
            # we save nuclei masks in a separate file to speed up features extraction without mask.
            if not args.box_only and 'masks' in outputs['cell_stats']:
                nuclei_masks = outputs['cell_stats']['masks']
                torch.save(nuclei_masks, res_file_masks)
            if 'masks' in outputs['cell_stats']:
                del outputs['cell_stats']['masks']
            outputs['meta_info'] = meta_info
            outputs['slide_info'] = dataset.slide.info()
            outputs['slide_size'] = dataset.slide_size
            outputs['rois'] = dataset.masks
            torch.save(outputs, res_file)

            print(f"Total time: {time.time()-t0} s")

        if args.save_img:
            img_file = os.path.join(args.output_dir, f"{slide_id}.png")
            if not os.path.exists(img_file):
                print(f"Exporting result to image: ", end="")
                t0 = time.time()
                outputs = torch.load(res_file) if not outputs else outputs
                mask = export_detections_to_image(
                    outputs['cell_stats'], outputs['slide_size'], 
                    labels_color=outputs['meta_info']['labels_color'],
                    save_masks=not args.box_only, border=3,
                )
                Image.fromarray(mask).save(img_file)
                print(f"{time.time()-t0} s")
        
        if args.save_csv:
            csv_file = os.path.join(args.output_dir, f"{slide_id}.csv")
            if not os.path.exists(csv_file):
                print(f"Exporting result to csv: ", end="")
                t0 = time.time()
                outputs = torch.load(res_file) if not outputs else outputs
                if args.export_text and 'labels_text' in outputs['meta_info']:
                    labels_text = outputs['meta_info']['labels_text']
                else:
                    labels_text = None
                df = export_detections_to_table(
                    outputs['cell_stats'], 
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
    parser.add_argument('--model_path', default=MODEL_PATH, type=str, help="Model path, torch jit model." )
    parser.add_argument('--output_dir', default='slide_results', type=str, help="Output folder.")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], type=str, help='Run on cpu or gpu.')
    parser.add_argument('--roi', default='tissue', type=str, help='ROI region.')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of batch size.')
    parser.add_argument('--num_workers', default=64, type=int, help='Number of workers for data loader.')
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
