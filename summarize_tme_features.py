import os
import time
import pickle
import argparse
import glob
from collections import Counter
from utils_features import *


SEED = 42
SCALE_FACTOR = 32
DEFAULT_MPP = 0.25
CLASSES = ['tumor', 'stroma', 'lympho']


def main(args):
    if args.data_path is not None:
        if os.path.isdir(args.data_path):
            slide_files = [_ for _ in os.listdir(args.data_path) if not _.startswith('.') and _.endswith('.svs')]
        else:
            slide_files = [args.data_path]
    else:
        if os.path.isdir(args.model_res_path):
            slide_files = [_ for _ in os.listdir(args.model_res_path) if not _.startswith('.') and _.endswith('.pt')]
        else:
            slide_files = [args.model_res_path]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    pkl_filename = os.path.join(args.output_dir, "nuclei_features.pkl")
    csv_filename = os.path.join(args.output_dir, "nuclei_features.csv")
    if os.path.exists(pkl_filename):
        with open(pkl_filename, 'rb') as f:
            outputs = pickle.load(f)
    else:
        outputs = {}

    if args.save_images:
        img_foldername = os.path.join(args.output_dir, "images")
        if not os.path.exists(img_foldername):
            os.makedirs(img_foldername)

    device = torch.device(args.device)
    for slide_file in slide_files:
        slide_id = os.path.splitext(slide_file)[0]
        print("==============================")
        print(slide_id)
        svs_file = os.path.join(args.data_path, f"{slide_id}.svs")
        if os.path.exists(svs_file):
            from utils_image import Slide
            try:
                slide = Slide(svs_file, verbose=False)
            except:
                slide = None
        
        if slide is not None:
            print(f"Find original slide: slide_id={slide.slide_id}, magnitude={slide.magnitude}, mpp={slide.mpp}, slide_size={slide.level_dims[0]}")
            slide_size = (slide.level_dims[0][1], slide.level_dims[0][0])
            slide_img = slide.thumbnail((1024, 1024))
        else:
            slide_size = None
            slide_img = None

        if slide_id in outputs and len(outputs[slide_id]):  # exists and not empty
            for k, v in outputs[slide_id].items():
                if v is None:
                    print((k, v))
            print({k: (v if isinstance(v, numbers.Number) else len(v))
                   for k, v in outputs[slide_id].items()})
            print("==============================")
            continue

        # res_file = glob.glob(os.path.join(args.model_res_path, slide_id, "cell_summary_*.csv"))[0]
        res_file = os.path.join(args.model_res_path, f"{slide_id}.pt")
        res_file_masks = os.path.join(args.model_res_path, f"{slide_id}.masks.pt")
        if os.path.exists(res_file):
            t0 = time.time()
            ## Yolo output use this
            res = torch.load(res_file)
            if not len(res['cell_stats']):
                print(f"Warning: {res_file} is empty!")
                outputs[slide_id] = {}
                continue
            res_nuclei = torch.cat([
                res['cell_stats']['boxes'], 
                res['cell_stats']['scores'][..., None], 
                res['cell_stats']['labels'][..., None]], -1
            )
            # [res['rois'], res['slide_size'], res['slide_info'], slide['meta_info']]
            mpp, slide_size = res['slide_info']['mpp'], res['slide_size']

            nuclei_map, r_ave = res2map_yolo_result(
                res_nuclei, slide_size=slide_size, n_classes=args.n_classes, 
                use_scores=False, scale=mpp/DEFAULT_MPP,
            )
            print(nuclei_map.shape, r_ave)

            ## Extract base features
            scatter_img = scatter_plot(nuclei_map, r_ave, scale_factor=1./args.scale_factor)
            base_features, cloud_d, roi_mask = extract_base_features(
                nuclei_map, r_ave, 
                scale_factor=1./args.scale_factor, 
                roi_indices=[0],
                n_patches=args.n_patches,
                patch_size=args.patch_size,
                nms_thresh=args.nms_thresh,
                score_thresh=args.score_thresh,
                seed=SEED, device=device,
            )
            density_img = density_plot(cloud_d, scale_factor=1./args.scale_factor)

            outputs[slide_id] = {
                'nuclei_map': nuclei_map, 'r_ave': r_ave, 
                'base_features': base_features, 'cloud_d': cloud_d, 
                'slide_img': slide_img, 'scatter_img': scatter_img, 
                'density_img': density_img, 'roi_mask': roi_mask, 
            }

            if args.save_images:
                if slide_img is not None:
                    save_path = os.path.join(img_foldername, f"{slide_id}.slide_img.png")
                    skimage.io.imsave(save_path, (slide_img*255.0).astype(np.uint8))

                save_path = os.path.join(img_foldername, f"{slide_id}.scatter_img.png")
                skimage.io.imsave(save_path, (scatter_img*255.0).astype(np.uint8))

                save_path = os.path.join(img_foldername, f"{slide_id}.density_img.png")
                skimage.io.imsave(save_path, (density_img*255.0).astype(np.uint8))

                save_path = os.path.join(img_foldername, f"{slide_id}.roi_mask.png")
                skimage.io.imsave(save_path, (roi_mask*255.0).astype(np.uint8))

            print({k: (v if isinstance(v, numbers.Number) else len(v))
                   for k, v in outputs[slide_id]['base_features'].items()})
            t3 = time.time()
            print(f"total time: {t3-t0} s.")
        else:
            print(f"Warning: {res_file} doesn't exists!")
            outputs[slide_id] = {}
        print("==============================")

        with open(pkl_filename, 'wb') as f:
            pickle.dump(outputs, f)

    ## Summarize normalized features
    if args.slides_mapping_file is not None and os.path.exists(args.slides_mapping_file):
        slide_pat_map = pd.read_csv(args.slides_mapping_file).to_dict()
    else:
        slide_pat_map = None

    bfs = {slide_id: entries['base_features'] for slide_id, entries in outputs.items()}
    df = summarize_normalized_features(bfs, slide_pat_map=slide_pat_map)
    df.to_csv(csv_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WSI feature extraction.', add_help=True)
    parser.add_argument('--model_res_path', required=True, default="./slide_results", type=str, 
                        help="The detection/segmentation result folder." )
    parser.add_argument('--output_dir', default="./feature_results", type=str, help="output folder.")
    parser.add_argument('--data_path', default='./', type=str, help="input data filename or directory.")
    parser.add_argument('--slides_mapping_file', default=None, type=str, 
                        help="A csv file explains slide_id -> patient_id.")
    parser.add_argument('--scale_factor', default=SCALE_FACTOR, type=float, help='1/scale factor for density analysis.')
    parser.add_argument('--n_classes', default=len(CLASSES), type=int, help='Number of nuclei types in analysis.')
    parser.add_argument('--n_patches', default=10, type=int, help='Number of maximum patches to analysis.')
    parser.add_argument('--patch_size', default=2048, type=int, help='Patch size for analysis.')
    parser.add_argument('--nms_thresh', default=0.015, type=float, help='maximum overlapping between patches.')
    parser.add_argument('--score_thresh', default=160, type=float, help='minimum coverage of tumor region.')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], type=str, 
                        help='Run density analysis on cpu or gpu.')
    parser.add_argument('--num_workers', default=32, type=int, help='Number of workers for density data loader.')
    parser.add_argument('--save_images', action='store_true', help='store img, dots, densities, roi masks.')
    
    args = parser.parse_args()
    
    main(args)
    