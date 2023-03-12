import math
import time
import torch
import torchvision
import torch.nn.functional as F

import numbers
import numpy as np
import pandas as pd
import scipy
import scipy.spatial

import cv2
import itertools
# import multiprocessing as mp
from functools import reduce
from collections import Counter
from scipy.signal import fftconvolve, convolve

import matplotlib
import skimage.morphology

from .utils_image import random_sampling_in_polygons, binary_mask_to_polygon, polygon_areas, ObjectProperties


""" Nuclei feature codes:
    1. Nuclei morphological features.
    roi_area: overall tumor/tissue region.
    i.radius: average radius for nuclei type_i base on bounding bx size.
    i.total: total amount of nuclei type_i detected in results file.
    i.box_area.*: statistics of bounding box area of type_i.
    i.area.*: statistics of mask area of type_i.
    i.convex_area.*: statistics of mask convex_area of type_i.
    i.eccentricity.*: statistics of mask eccentricity of type_i.
    i.extent.*: statistics of mask extent of type_i.
    i.filled_area.*: statistics of mask filled_area of type_i.
    i.major_axis_length.*: statistics of mask major_axis_length of type_i.
    i.minor_axis_length.*: statistics of mask minor_axis_length of type_i.
    i.orientation.*: statistics of mask orientation of type_i.
    i.perimeter.*: statistics of mask perimeter of type_i.
    i.solidity.*: statistics of mask solidity of type_i.
    i.pa_ratio.*: statistics of mask pa_ratio (perimeter ** 2 / filled_area) of type_i.

    2. Delaunary graph based features: select (maximum 10) random patches (default 2048*2048) in tumor region.
    i.count: total No. of nuclei type_i in selected patches (different from whole slide level i.total).
    i.count.prob: percentage of nuclei type_i in selected patches.
    i_j.edges.mean: average nuclei distance of type_i <-> type_j interactions.
    i_j.edges.std: nuclei distances std of type_i <-> type_j interactions.
    i_j.edges.count: total No. of type_i <-> type_j interactions in all patches.
    i_j.edges.marginal.prob: i_j.edges.count/sum(x_y.edges.count), percentage of type_i <-> type_j interaction.
    i_j.edges.conditional.prob: i_j.edges.count/sum(x_j.edges.count), edge probability condition to type_j interaction.
    i_j.edges.dice: dice coefficient of i_x.edges and y_j.edges, overlap over union of type_i interaction and type_j interaction.

    3. Density based features: use kernel smoothing to transfer nuclei into density map.
    i_j.dot: dot product between type_i and type_j.
    i.norm: norm2(type_i), type_i density.
    i_j.proj: i_j.dot / j.norm, influence of type_i on type_j.
    i_j.proj.logit: log scale of i_j.proj, make support for add/minus between features. 
    i_j.proj.prob: use sigmoid activation to transfer i_j.proj.logit into probability score (0~1).
    i_j.cos: i_j.dot/i.norm/j.norm, similarity of type_i and type_j.
"""

TO_REMOVE = 1
# OBJECT_FEATURES = ['coordinate_x', 'coordinate_y', 'cell_type', 'probability']
REGIONPROP_FEATURES = {
    'area': lambda prop: prop.area, 
    'convex_area': lambda prop: prop.convex_area, 
    'eccentricity': lambda prop: prop.eccentricity, 
    'extent': lambda prop: prop.extent,
    'filled_area': lambda prop: prop.filled_area, 
    'major_axis_length': lambda prop: prop.major_axis_length, 
    'minor_axis_length': lambda prop: prop.minor_axis_length,
    'orientation': lambda prop: prop.orientation, 
    'perimeter': lambda prop: prop.perimeter, 
    'solidity': lambda prop: prop.solidity,
    'pa_ratio': lambda prop: 1.0 * prop.perimeter ** 2 / prop.filled_area, 
}


def filter_wsi_results(x, n_classes=None, **kwargs):
    """ Remove unused/unclassified labels, drop low scores and remove occlusion. """
    score_threshold = kwargs.get('score_threshold', 0.)
    iou_threshold = kwargs.get('iou_threshold', 1.)
    n_classes = n_classes or int(x['labels'].max().item())

    keep = (x['labels'] <= n_classes) & (x['labels'] >= 0)
    x = {k: v[keep] for k, v in x.items()}
    if score_threshold > 0.:
        keep = x['scores'] >= score_threshold
        x = {k: v[keep] for k, v in x.items()}
    if iou_threshold < 1.:
        keep = torchvision.ops.nms(x['boxes'], x['scores'], iou_threshold=iou_threshold)
        x = {k: v[keep] for k, v in x.items()}

    return x


def generate_nuclei_map(x, slide_size=None, n_classes=None, use_scores=False):
    n_classes = n_classes or int(x['labels'].max().item())
    if slide_size is None:
        h, w = int(math.ceil(x['boxes'][:,[1,3]].max().item())+1), int(math.ceil(x['boxes'][:,[0,2]].max().item())+1)
    else:
        h, w = slide_size

    # calculate average size for each class
    d = ((x['boxes'][:,3]-x['boxes'][:,1]) * (x['boxes'][:,2]-x['boxes'][:,0])) ** 0.5
    r_ave = {}
    for _ in range(n_classes): # 0.median() gives segmentation fault...
        tmp = d[x['labels'] == _+1]
        r_ave[_] = tmp.median().item()/2 if len(tmp) else np.nan

    # get coordinates
    i_c = x['labels'] - 1
    i_x = ((x['boxes'][:,1] + x['boxes'][:,3])/2).round()
    i_y = ((x['boxes'][:,0] + x['boxes'][:,2])/2).round()
    val = x['scores']
    # build point clouds
    pts = torch.sparse_coo_tensor([i_c.tolist(), i_x.tolist(), i_y.tolist()], 
                                  (val if use_scores else [1.0] * len(val)), 
                                  (n_classes, h, w)).coalesce()

    return pts, r_ave


def rescale_nuclei_map(x, radius, scale_factor=1.0, grid=4096):
    if scale_factor == 1.0:  # no rescale, to_dense()
        return x.to_dense()[list(radius.keys())]

    n_classes, h0, w0 = x.shape
    h, w = (round(h0*scale_factor), round(w0*scale_factor))
    if h0 < grid and w0 < grid:  # for small slide: h, w < grid
        x = x.to_dense()[list(radius.keys())]
        x = F.adaptive_avg_pool2d(x[None], (h, w))[0]
        # x = F.interpolate(x[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
        return x
    else:  # for large slide, can't to_dense() into memory
        values, indices = x.values(), x.indices()
        patches = []
        for i in range(math.ceil(h0/grid)):
            row = []
            for j in range(math.ceil(w0/grid)):
                x0, x1 = i * grid, (i+1) * grid
                y0, y1 = j * grid, (j+1) * grid
                keep = (indices[1] >= x0) & (indices[1] < x1) & (indices[2] >= y0) & (indices[2] < y1)
                # (np.isin(indices[0].numpy(), list(r_ave.keys())))
                s_values, s_indices = values[keep], indices[:,keep]

                s_indices[1] -= x0
                s_indices[2] -= y0
                pts = torch.sparse_coo_tensor(s_indices, s_values, (n_classes, grid, grid))
                pts = pts.to_dense()[list(radius.keys())]
                row.append(F.adaptive_avg_pool2d(pts[None], (round(grid*scale_factor), round(grid*scale_factor)))[0])
            patches.append(torch.cat(row, 2))
        patches = torch.cat(patches, 1)

        return patches[:, :h, :w]


def scatter_plot(nuclei_map, r_ave, labels_color, scale_factor=1./64):
    nc, h, w = nuclei_map.shape
    c, x, y = nuclei_map.indices()
    values = nuclei_map.values()

    # color tensor
    color_tensor = torch.zeros((nc + 1 + 1, 4), dtype=torch.float32)
    for k, v in labels_color.items():
        if k > 0 and k <= nc:
            color_tensor[k] = torch.tensor(matplotlib.colors.to_rgba(v))
        else:
            color_tensor[-1] = torch.tensor(matplotlib.colors.to_rgba(v))

    i_c = torch.tensor([0,1,2,3]).repeat(len(c))
    i_x = x.repeat_interleave(4)
    i_y = y.repeat_interleave(4)
    val = F.embedding(c+1, color_tensor).flatten()  # c start from 0, labels_color start with 1

    new_pts = torch.sparse_coo_tensor([i_c.tolist(), i_x.tolist(), i_y.tolist()], val, 
                                      (4, h, w)).coalesce()
    r_rgb = torch.stack([color_tensor[k+1] * v for k, v in r_ave.items()]).numpy()
    r_rgb = {k: v for k, v in enumerate(np.nanmax(r_rgb, 0))}
    pts_image = rescale_nuclei_map(new_pts, r_rgb, scale_factor=scale_factor, grid=4096)
    p_image = (pts_image.permute(1, 2, 0).numpy() > 0) * 1.0

    return p_image


def density_plot(cloud_d, scale_factor):
    # d_map = torch.stack([cloud_d[1]/cloud_d[1].max(), cloud_d[0]/cloud_d[0].max()/1.5, cloud_d[2]/cloud_d[2].max()]) * 3
    d_map = torch.stack([cloud_d[1], cloud_d[0], cloud_d[2]]) / scale_factor * 3
    # d_map = torch.nn.functional.interpolate(d_map[None], scale_factor=scale_factor)[0]
    # t_l = d_map[1] * d_map[2]
    d_map = d_map.permute(1, 2, 0).numpy()

    return np.clip(d_map, 0., 1.)


def apply_filters(x, radius, scale_factor=1.0, method='gaussian', grid=4096, device=torch.device('cpu')):
    ## shrink image
    x_rescale = rescale_nuclei_map(x, radius, scale_factor=scale_factor, grid=grid)

    res = {}
    for k, r in radius.items():
        if np.isnan(r) or r <= 1.:
            res[k] = x_rescale[k]
        else:
            r = r * scale_factor * 8
            if method == 'gaussian':
                d, c = round(2 * r), round(r)
                n = np.zeros((2*d+1, 2*d+1))
                n[d-c:d+c+1, d-c:d+c+1] = skimage.morphology.disk(c)
                kernel = scipy.ndimage.gaussian_filter(n, sigma=c/2) * skimage.morphology.disk(d)
            elif method == 'mean':
                kernel = skimage.morphology.disk(round(2*r))
    #         elif method == 'edt':
    #             d = round(2 * r)
    #             n = np.zeros((2*d+1, 2*d+1))
    #             kernel = skimage.morphology.disk(round(r))
    #             kernel = scipy.ndimage.distance_transform_edt(kernel)

            try:  # RuntimeError: std::bad_alloc with torch 1.9 and 1.10 on cpu
                padding = [_//2 for _ in kernel.shape]
                m = F.conv2d(x_rescale[k][None, None].to(device), 
                             weight=torch.as_tensor(kernel, dtype=torch.float32).to(device)[None, None], 
                             padding=padding, bias=None)[0, 0].cpu()
            except:
                print(f"Use scipy for convolution.")
                m = torch.as_tensor(convolve(x_rescale[k], kernel, mode='same', method='direct'), dtype=torch.float32)
            res[k] = m

    return x_rescale, res


def product_features(x):
#     res = {_: torch.norm(x[_].flatten(), p=2).item() ** 2 for _ in range(len(x))}
#     for i, j in itertools.combinations(range(len(x)), 2):
#         res['{}_{}'.format(i, j)] = torch.dot(x[i].flatten(), x[j].flatten()).item()

    # x is a dict of {'type_idx': density_plot}
    res = {f'{i}_{j}.dot': torch.dot(x[i].flatten(), x[j].flatten()).item()
           for i, j in itertools.combinations_with_replacement(x, 2)}

    return res


def delaunay_features(coords, labels, min_dist=0., max_dist=np.inf):
    tri = scipy.spatial.Delaunay(coords)
    # indices, indptr = tri.vertex_neighbor_vertices
    indices = tri.simplices.T

    # merge results
    idx_1 = indices.flatten()
    idx_2 = indices[[1,2,0]].flatten()
    pairs = np.array(['{}_{}'.format(*sorted(_)) for _ in zip(labels[idx_1], labels[idx_2])])
    dists = torch.norm((coords[idx_1] - coords[idx_2]), p=2, dim=1).numpy()
    keep = (dists > min_dist) * (dists < max_dist)

    return pairs[keep].tolist(), dists[keep].tolist()


def polygons2mask(polygons, shape, scale=1.):
    """ Use cv2.fillPoly to be consistent with (w, h) pattern.
    """
    mask = np.zeros((shape[1], shape[0]))
    polygons = [((np.array(_) * scale)).astype(np.int32) 
                for _ in polygons]
    cv2.fillPoly(mask, polygons, 1)

    return mask.astype(bool)


def random_patches(N, patch_size, polygons, image_size=None, scores_fn=None,
                   nms_threshold=0.3, score_threshold=0.6, sampling_factor=10,
                   seed=None, plot_selection=False):
    # from nms.nms import boxes as nms_boxes
    if isinstance(patch_size, numbers.Number):
        patch_size = (patch_size, patch_size)
    patch_size = np.array(patch_size)

    if image_size is None:
        w, h = np.concatenate(polygons).max(0).astype(np.int32)
    else:
        w, h = image_size

    pool_size = N * sampling_factor
    coords, indices = random_sampling_in_polygons(polygons, pool_size, plot=plot_selection, seed=seed)
    coords = coords - patch_size / 2 + np.random.uniform(-0.5, 0.5, size=(pool_size, 2)) * patch_size
    ## shift bboxes at corner and border into valid region, flip x and y to match cv2 functions
    coords[:,0] = np.clip(coords[:,0], 0, w - patch_size[0])
    coords[:,1] = np.clip(coords[:,1], 0, h - patch_size[1])
    patches = np.hstack([coords.astype(np.int), np.tile(patch_size, (pool_size, 1))])

    ## generate mask and calculate iou (on the lowest resolution to save time)
    if scores_fn is None:  # use mask_region/patch_size as score
        # masks = self.polygons2mask(polygons, shape=self.level_dims[-1], scale=1./scales[-1])
        # (w, h), scales = self.get_scales(image_size)
        masks = polygons2mask(polygons, shape=(w, h))
        # skimage.io.imsave('./masks_poly.png', masks)
        scores = np.array([masks[y0:y0+dh, x0:x0+dw].sum()/dw/dh for x0, y0, dw, dh in patches.astype(np.int)])
    else:
        scores = np.array([scores_fn(_) for _ in patches.astype(np.int)])

    cutoff = scores >= score_threshold
    patches, scores, indices = patches[cutoff], scores[cutoff], indices[cutoff]
    patches = torch.from_numpy(patches).type(torch.float32)
    scores = torch.from_numpy(scores).type(torch.float32)
    keep = torchvision.ops.nms(patches, scores, nms_threshold)[:N]
#     # cv2 cause segmentation fault...
#     keep = cv2.dnn.NMSBoxes(patches.tolist(), scores.tolist(), score_threshold=score_threshold, 
#                             nms_threshold=nms_threshold, eta=0.9, top_k=N)
#     keep = keep.flatten() if len(keep) else []

    return patches[keep], indices[keep]


def _regionprops(box, label, mask):
    box = box.round().to(torch.int32)
    w = max((box[2] - box[0] + TO_REMOVE).item(), 1)
    h = max((box[3] - box[1] + TO_REMOVE).item(), 1)
    o = {'box_area': h * w, 'labels': label.item()}

    mask = cv2.resize(mask[0].float().numpy(), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
    if mask.sum():
        prop = ObjectProperties(box.numpy(), mask)
        o.update({k: fn(prop) for k, fn in REGIONPROP_FEATURES.items()})
    else:
        o.update({k: np.nan for k, fn in REGIONPROP_FEATURES.items()})

    return o


def extract_nuclei_features(x, box_only=False, n_classes=None, num_workers=None, **kwargs):
    """ Extract features from detection results.
        All x, y follow skimage sequence. (opposite to PIL).
        x: row, height. y: col, width.
    """
    n_classes = n_classes or int(x['labels'].max().item())
    if box_only or 'masks' not in x:
        w = (x['boxes'][:,2] - x['boxes'][:,0] + TO_REMOVE)
        h = (x['boxes'][:,3] - x['boxes'][:,1] + TO_REMOVE)
        return {'box_area': (h * w).numpy(), 'labels': x['labels'].numpy()}
    else:
        # we need a way to vectorize this, pool.starmap is super slow
        df = [_regionprops(box, label, mask)
              for box, label, mask in zip(x['boxes'], x['labels'], x['masks'])]
#         with mp.Pool(num_workers) as pool:
#             df = pool.starmap(_regionprops, zip(x['boxes'], x['labels'], x['masks']))

        return pd.DataFrame(df).to_dict(orient='list')


def extract_tme_features(nuclei_map, radius, scale_factor=0.1, roi_indices=[0],
                          n_patches=10, patch_size=2048, nms_thresh=0.3, score_thresh=0.8, 
                          max_dist=100., seed=42, device=torch.device('cpu')):
    """ Extract TME features from slides. 
        nuclei_map is a sparse matrix, radius is a dictionary records {type_idx: type_r}, 
        remove type_idx from radius if this class doesn't exists or doesn't want to be used 
        in feature extraction (Note that: removing categories will change the denularity graph 
        and may influence results. Exp: a tumor region full with lots of necrosis or blood cell.)
        (All TME features in this function are addable/concatable if merging is needed.
         Merge these TME features before compute addon features. See compute_normalized_features. )
    """
    ## Apply kernel density, cloud_d will only keep type_idx exists in radius
    _, cloud_d = apply_filters(nuclei_map, radius, scale_factor=scale_factor, 
                               method='gaussian', grid=4096, device=device)  # int(1024/scale_factor)
    dot_product = product_features(cloud_d)  # dot products
    nuclei_radius = {f'{k}.radius': v for k, v in radius.items()}

    ## Extract ROI based on nuclei type, default roi_indices=[0], means extract nuclei_type=0 (tumor region).
    scale_mask = 10 ** int(2-math.log10(min(cloud_d[0].shape)))    # scale_mask = 0.25
    masks = F.interpolate(torch.stack([cloud_d[_] for _ in roi_indices])[None], 
                          scale_factor=scale_mask, mode='bilinear', 
                          align_corners=False, recompute_scale_factor=False).sum(0)
    masks = scipy.ndimage.binary_fill_holes(torch.any(masks > 0, dim=0).numpy())
    # masks = skimage.filters.gaussian(masks, sigma=5) > 0.1

    ## Random select patches from ROI
    scale_poly = scale_factor * scale_mask  # scale_poly=scale_factor*0.25, 20x slide will have half
    polygons = binary_mask_to_polygon(masks, mode='yx', scale=1./scale_poly)
    poly_areas = [polygon_areas(_) for _ in polygons]
    polygons = [_ for _, area in zip(polygons, poly_areas) if area / sum(poly_areas) >= 0.05]

    ## extract nuclei counts info and delaunay graph info
    cellinfo = {'counts': []}
    delaunay = {'pairs': [], 'dists': []}
    if polygons:
        ## extract delanunay features and counts
        labels, x, y = nuclei_map.indices()
        coords = torch.stack([x, y], dim=-1).float()

        def _score_fn(patch):
            x0, y0, w, h = patch
            keep = (coords[:,0] >= y0) & (coords[:,1] >= x0) & (coords[:,0] < y0+w) & (coords[:,1] < x0+h)
            coords_patch, labels_patch = coords[keep], labels[keep]
            core_nuclei = sum(labels_patch==i for i in roi_indices)

            return core_nuclei.sum()

        # random select patches
        patches, indices = random_patches(
            n_patches * 2, patch_size, polygons, 
            scores_fn=_score_fn,
            nms_threshold=nms_thresh, 
            score_threshold=score_thresh, 
            seed=seed, plot_selection=False,
        )

        k_patch = 0
        for x0, y0, w, h in patches:
            if k_patch == n_patches:
                break
            keep = (coords[:,0] >= y0) & (coords[:,1] >= x0) & (coords[:,0] < y0+w) & (coords[:,1] < x0+h)
            coords_patch, labels_patch = coords[keep], labels[keep]
            core_nuclei = sum(labels_patch==i for i in roi_indices)

            if keep.sum() < 10 or core_nuclei.sum() < 10:
                print(f"Warning: patch: [{x0}, {y0}, {x0+h}, {y0+w}] is ignored, less than 10 nuclei in this patch.")
                continue
            else:
                print(f"Analyzed patch: [{x0}, {y0}, {x0+h}, {y0+w}], with {core_nuclei.sum()} tumor inside.")

            pairs_patch, dists_patch = delaunay_features(coords_patch, labels_patch, max_dist=max_dist)
            k_patch += 1
            delaunay['pairs'].extend(pairs_patch)
            delaunay['dists'].extend(dists_patch)
            cellinfo['counts'].extend(labels_patch.numpy())
        print(f"Analyzed {k_patch} randomly selected patches (size={patch_size}).")
    else:
        print(f"Warning: can't find valid polygons (>=0.05*roi_area) in masks. ")

    cell_counts = Counter(cellinfo['counts'])
    cellinfo_f = {f'{k}.count': cell_counts.get(k, 0) for k in radius}

    return {'roi_area': sum(poly_areas), **nuclei_radius, **cellinfo_f, **dot_product, **delaunay,}, cloud_d, masks


def merge_tme_features_by_ids(x, slide_pat_map=None):
    """ Add (numbers) or Concat (lists) for dictionary. 
        slide_pat_map={'slide_id' : 'merged_pat_id'}
    """
    if slide_pat_map is None:
        return x

    def dict_add(x, y):
        res = {}
        for k in x.keys() | y.keys():
            if k in x and k in y:
                res[k] = x[k] + y[k]
            elif k in x:
                res[k] = x[k]
            elif k in y:
                res[k] = y[k]

        return res

    data = defaultdict(list)
    for slide_id, pat_id in slide_pat_map.items():
        data[pat_id].append(x[slide_id])

    return {pat_id: reduce(dict_add, _) for pat_id, _ in data.items()}


def name_converter(x, rep):
    """ Used to replace index with name in feature names. 
        rep={'0': 't', '1': s, '2': l, ...}
        '0.count' -> 't.count', '0_1.prob' -> 't_s/prob'
    """
    # pattern = re.compile("|".join(rep.keys()))
    # return pattern.sub(lambda m: rep[re.escape(m.group(0))], a)
    entry = x.split('.')
    entry[0] = '_'.join([rep.get(_, _) for _ in entry[0].split('_')])
    return '.'.join(entry)


def summarize_normalized_features(x, slide_pat_map=None):
    """ Run extract_tme_features first,
        Then run this command to generate normalized features:
        probility, dice, iou, projection, cosine similarity etc.
        If slides need to be merged together based on patient id,
        provide a dictionary: slide_pat_map {slide_id: pat_id}, script will add/concat 
        all the slots from the results in extract_tme_features, then normalize features. 
    Args: 
        x (dict): slide_id/pat_id -> tme_features
    """
    ## Merge slides if necessary
    x = merge_tme_features_by_ids(x, slide_pat_map)

    ## Calculate mean, std, count for edges, merge all features into dataframe
    df = {}
    for slide_id, entry in x.items():
        nuclei_fnames = ['box_area'] + [k for k in REGIONPROP_FEATURES if k in entry]
        nuclei = pd.DataFrame.from_dict({k: entry[k] for k in ['labels'] + nuclei_fnames}, orient='columns')
        nuclei_f = nuclei.groupby('labels').agg(['mean', 'std'])
        nuclei_f.columns = [f'{k}.{tag}' for k, tag in nuclei_f.columns]
        nuclei_f = nuclei_f.stack()
        nuclei_f.index = [f'{k}.{v}' for k, v in nuclei_f.index]
        nuclei_f = {
            **{f'{k}.total': v for k, v in Counter(nuclei['labels']).items()},
            **nuclei_f.to_dict(),
        }

        if 'pairs' in entry and 'dists' in entry:
            edges = pd.DataFrame.from_dict({'pairs': entry['pairs'], 'dists': entry['dists']}, orient='columns')
            edges_f = edges.groupby('pairs').agg(['mean', 'std', 'count'])['dists']
            edges_f = {f"{_['pairs']}.edges.{_['variable']}": _['value'] 
                       for idx, _ in edges_f.reset_index().melt(id_vars=['pairs']).iterrows()}
        else:
            edges_f = {}
        # ugly way to add missing entries in edges_f
        class_ids = [_.split('.')[0] for _ in entry.keys() if _.endswith('radius')]
        for i, j in itertools.combinations_with_replacement(class_ids, 2):
            i, j = min(i, j), max(i, j)
            edges_f[f'{i}_{j}.edges.mean'] = edges_f.get(f'{i}_{j}.edges.mean', np.nan)
            edges_f[f'{i}_{j}.edges.std'] = edges_f.get(f'{i}_{j}.edges.std', np.nan)
            edges_f[f'{i}_{j}.edges.count'] = edges_f.get(f'{i}_{j}.edges.count', 0.)

        # df[slide_id] = {'id': slide_id, **edges_f, **{k: v for k, v in entry.items() if isinstance(v, numbers.Number)},}
        df[slide_id] = {
            'id': slide_id, **nuclei_f, **edges_f, 
            **{k: v for k, v in entry.items() if isinstance(v, numbers.Number)},
        }

    df = pd.DataFrame.from_dict(df, orient='index').set_index('id')  # from_dict will ignore empty entry...
    class_ids = sorted([int(_.split('.')[0]) for _ in df.columns if _.endswith('.radius')])

    ## Continuous features:
    sigmoid = lambda x: 1 / (1 + np.exp(-x))  # sigmoid function to normalize logit
    # norm: 
    for idx in class_ids:
        df[f'{idx}.norm'] = df[f'{idx}_{idx}.dot'] ** 0.5
        df[f'{idx}.norm.logit'] = np.log(df[f'{idx}.norm'])

    # projection (direction): df['i_j.proj'] = df['i_j.dot'] / df['j.dot']
    for i, j in itertools.permutations(class_ids, 2):
        df[f'{i}_{j}.proj'] = df[f'{min(i, j)}_{max(i, j)}.dot'] / df[f'{j}_{j}.dot']
        df[f'{i}_{j}.proj.logit'] = np.log(df[f'{i}_{j}.proj'])
        df[f'{i}_{j}.proj.prob'] = sigmoid(df[f'{i}_{j}.proj.logit'])

    # cosine similarity: df['i_j.cos'] = df['j_j.dot']/df['i.norm']/df['j.norm']
    for i, j in itertools.combinations(class_ids, 2):
        i, j = min(i, j), max(i, j)
        df[f'{i}_{j}.cos'] = df[f'{i}_{j}.dot']/df[f'{i}.norm']/df[f'{j}.norm']

    ## Discrete features:
    # nuclei count percentage:
    total_counts = df[[f'{idx}.count' for idx in class_ids]].sum(1)
    for idx in class_ids:
        df[f'{idx}.count.prob'] = df[f'{idx}.count']/total_counts

    # edge marginal percentage: edge_i_j.count/total_edge
    total_edges = df[[f'{i}_{j}.edges.count' for i, j in itertools.combinations_with_replacement(class_ids, 2)]].sum(1)
    for i, j in itertools.combinations_with_replacement(class_ids, 2):
        df[f'{i}_{j}.edges.marginal.prob'] = df[f'{i}_{j}.edges.count'] / total_edges

    # edge conditional probabilities: edge_i_j.count/total_edge_j.count
    for i, j in itertools.permutations(class_ids, 2):
        total_edges_j = df[[f'{min(_, j)}_{max(_, j)}.edges.count' for _ in class_ids]].sum(1)
        df[f'{i}_{j}.edges.conditional.prob'] = df[f'{min(i, j)}_{max(i, j)}.edges.count']/total_edges_j

    # dice-coefficient: 2*edge_i_j.count/(total_edge_i.count + total_edge_j.count)
    for i, j in itertools.combinations(class_ids, 2):
        total_edges_i = df[[f'{min(_, i)}_{max(_, i)}.edges.count' for _ in class_ids]].sum(1)
        total_edges_j = df[[f'{min(_, j)}_{max(_, j)}.edges.count' for _ in class_ids]].sum(1)
        df[f'{i}_{j}.edges.dice'] = 2 * df[f'{i}_{j}.edges.count']/(total_edges_i + total_edges_j)

    return df
