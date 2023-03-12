import os

DEFAULT_MPP = 0.25
PATCH_SIZE = 512
PADDING = 64
PAGE = 0
MASK_ALPHA = 0.3

MODEL_PATHS = {
    'lung': './selected_models/benchmark_lung/lung_best.float16.torchscript.pt',
    'brca': './selected_models/benchmark_nucls_paper/fold3_epoch201.float16.torchscript.pt',
#     'nucls1': './selected_models/benchmark_nucls_paper/fold1_epoch6.float16.torchscript.pt',
#     'nucls2': './selected_models/benchmark_nucls_paper/fold2_epoch71.float16.torchscript.pt',
#     'nucls3': './selected_models/benchmark_nucls_paper/fold3_epoch201.float16.torchscript.pt',
#     'nucls4': './selected_models/benchmark_nucls_paper/fold4_epoch102.float16.torchscript.pt',
#     'nucls5': './selected_models/benchmark_nucls_paper/fold5_epoch127.float16.torchscript.pt',
}

DATASETS = {
    'patch_size': PATCH_SIZE,
    'padding': PADDING,
    'page': PAGE,
    'labels': ['bg', 'tumor', 'stromal', 'immune', 'blood', 'macrophage', 'dead', 'other',],
    'labels_color': {
        -100: '#949494',
        0: '#ffffff', 
        1: '#00ff00', 
        2: '#ff0000', 
        3: '#0000ff', 
        4: '#ff00ff', 
        5: '#ffff00',
        6: '#0094e1',
        7: '#646464',
    },
    'labels_text': {
        0: 'bg', 1: 'tumor', 2: 'stromal', 3: 'immune', 
        4: 'blood', 5: 'macrophage', 6: 'dead', 7: 'other',
    },
}

NMS_PARAMS = {
    'conf_thres': 0.15,  # score_threshold, discards boxes with score < score_threshold
    'iou_thres': 0.45,  # iou_threshold, discards all overlapping boxes with IoU > iou_threshold
    'classes': None, 
    'agnostic': True, # False
    'multi_label': False, 
    'labels': (), 
    'max_det': 1000,  # maximum detection
}

ROI_NAMES = {
    'tissue': True,  # use tissue region as roi
    'xml': '.*',  # use all annotations in slide_id.xml 
}

TIFF_PARAMS = {
    'tile': (256, 256), 
    'photometric': 'RGB',
    'compress': True,  # compression=('jpeg', 95),  # None RGBA, requires imagecodecs
}