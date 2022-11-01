#!/usr/bin/env python
#
# deepzoom_multiserver - Example web application for viewing multiple slides
#
# Copyright (c) 2010-2015 Carnegie Mellon University
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of version 2.1 of the GNU Lesser General Public License
# as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from collections import OrderedDict
from io import BytesIO
from optparse import OptionParser
import os
from threading import Lock

from flask import Flask, abort, make_response, render_template, url_for

if os.name == 'nt':
    _dll_path = os.getenv('OPENSLIDE_PATH')
    if _dll_path is not None:
        if hasattr(os, 'add_dll_directory'):
            # Python >= 3.8
            with os.add_dll_directory(_dll_path):
                import openslide
        else:
            # Python < 3.8
            _orig_path = os.environ.get('PATH', '')
            os.environ['PATH'] = _orig_path + ';' + _dll_path
            import openslide

            os.environ['PATH'] = _orig_path
else:
    import openslide

from openslide import OpenSlide, OpenSlideError
from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from run_patch_inference import analyze_one_patch
from utils_wsi import load_cfg, export_detections_to_image
import configs as CONFIGS
import torch
import tifffile
from torchvision.transforms import ToTensor


SLIDE_DIR = '.'
MASKS_DIR = '.'
SLIDE_CACHE_SIZE = 10
DEEPZOOM_FORMAT = 'png'
DEEPZOOM_TILE_SIZE = 254
DEEPZOOM_OVERLAP = 1
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 75
DEEPZOOM_MODEL = None

def run_tile(tile, model, dataset_configs, mpp=0.25, device=torch.device('cuda')):
    outputs = analyze_one_patch(
        ToTensor()(tile), model, dataset_configs, mpp=mpp, 
        compute_masks=True, device=device,
    )
    # cx = (outputs['cell_stats']['boxes'][:,0]+outputs['cell_stats']['boxes'][:,2])/2
    # cy = (outputs['cell_stats']['boxes'][:,1]+outputs['cell_stats']['boxes'][:,3])/2
    # keep = (cx >= DEEPZOOM_OVERLAP) & (cx < DEEPZOOM_OVERLAP+DEEPZOOM_TILE_SIZE) & (cy >= DEEPZOOM_OVERLAP) & (cy < DEEPZOOM_OVERLAP+DEEPZOOM_TILE_SIZE)
    # outputs['cell_stats'] = {k: v[keep] for k, v in outputs['cell_stats'].items()}

    # save image
    mask_img = export_detections_to_image(
        outputs['cell_stats'], tile.size, 
        labels_color=dataset_configs['labels_color'],
        save_masks=True, border=3,
        alpha=CONFIGS.MASK_ALPHA,
    )

    return mask_img


class DeepZoomGeneratorRGBA(DeepZoomGenerator):
    def get_tile(self, level, address):
        """ Keep RGBA alpha channel. """
        # Read tile
        args, z_size = self._get_tile_info(level, address)
        tile = self._osr.read_region(*args)

        # Scale to the correct size
        if tile.size != z_size:
            # Image.Resampling added in Pillow 9.1.0
            # Image.LANCZOS removed in Pillow 10
            tile.thumbnail(z_size, getattr(Image, 'Resampling', Image).LANCZOS)

        return tile


app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('DEEPZOOM_MULTISERVER_SETTINGS', silent=True)


class _SlideCache:
    def __init__(self, cache_size, dz_opts):
        self.cache_size = cache_size
        self.dz_opts = dz_opts
        self._lock = Lock()
        self._cache = OrderedDict()

    def get(self, slide_path, masks_path):
        with self._lock:
            if slide_path in self._cache:
                # Move to end of LRU
                slide, masks = self._cache.pop(slide_path)
                self._cache[slide_path] = (slide, masks)
                return slide, masks

        osr = OpenSlide(slide_path)
        slide = DeepZoomGenerator(osr, **self.dz_opts)
        try:
            mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
            slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            slide.mpp = 0
        
        if masks_path is not None and os.path.exists(masks_path):
            osr_masks = open_slide(masks_path)
            masks = DeepZoomGeneratorRGBA(osr_masks, **self.dz_opts)
        else:
            masks = None
        
        with self._lock:
            if slide_path not in self._cache:
                if len(self._cache) == self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[slide_path] = (slide, masks)
        return slide, masks


class _Directory:
    def __init__(self, slide_dir, relpath=''):
        self.name = os.path.basename(relpath)
        self.children = []
        for name in sorted(os.listdir(os.path.join(slide_dir, relpath))):
            cur_relpath = os.path.join(relpath, name)
            cur_path = os.path.join(slide_dir, cur_relpath)
            if os.path.isdir(cur_path):
                cur_dir = _Directory(slide_dir, cur_relpath)
                if cur_dir.children:
                    self.children.append(cur_dir)
            elif OpenSlide.detect_format(cur_path):
                self.children.append(_SlideFile(cur_relpath))


class _SlideFile:
    def __init__(self, relpath):
        self.name = os.path.basename(relpath)
        self.url_path = relpath


@app.before_first_request
def _setup():
    app.slide_dir = os.path.abspath(app.config['SLIDE_DIR'])
    app.masks_dir = os.path.abspath(app.config['MASKS_DIR'])
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = {v: app.config[k] for k, v in config_map.items()}
    app.cache = _SlideCache(app.config['SLIDE_CACHE_SIZE'], opts)
    
    # register device and model
    modelpath = app.config['DEEPZOOM_MODEL']
    app.device = torch.device('cuda')
    if modelpath in CONFIGS.MODEL_PATHS:
        modelpath = CONFIGS.MODEL_PATHS[modelpath]
    model = torch.jit.load(modelpath)
    model.eval()
    model.to(app.device)
    app.model = model

    meta_info = load_cfg('meta_info.yaml')
    app.dataset_configs = {'mpp': CONFIGS.DEFAULT_MPP, **CONFIGS.DATASETS, **meta_info}


def _get_slide(path):
    slide_path = os.path.abspath(os.path.join(app.slide_dir, path))
    masks_path = os.path.abspath(os.path.join(app.masks_dir, f"{os.path.splitext(path)[0]}.tiff"))
    if not slide_path.startswith(app.slide_dir + os.path.sep):
        # Directory traversal
        abort(404)
    if not os.path.exists(slide_path):
        abort(404)
    try:
        slide, masks = app.cache.get(slide_path, masks_path)
        slide.filename = os.path.basename(slide_path)
        return slide, masks
    except OpenSlideError:
        abort(404)


@app.route('/')
def index():
    return render_template('files.html', root_dir=_Directory(app.slide_dir))


@app.route('/<path:path>')
def slide(path):
    slide, masks = _get_slide(path)
    slide_url = url_for('dzi', path=path)
    return render_template(
        'slide-fullpage.html',
        slide_url=slide_url,
        slide_filename=slide.filename,
        slide_mpp=slide.mpp,
    )


@app.route('/<path:path>.dzi')
def dzi(path):
    slide, masks = _get_slide(path)
    format = app.config['DEEPZOOM_FORMAT']
    resp = make_response(slide.get_dzi(format))
    resp.mimetype = 'application/xml'
    return resp


@app.route('/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(path, level, col, row, format):
    slide, masks = _get_slide(path)
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = slide.get_tile(level, (col, row))
        if masks is not None:
            mask = masks.get_tile(level, (col, row))
        else:
            if level == slide._dz_levels-1:
                mask = run_tile(tile, app.model, app.dataset_configs, mpp=None, device=app.device)
                mask = Image.fromarray(mask)
            else:
                mask = None
    except ValueError:
        # Invalid level or coordinates
        abort(404)

    buf = BytesIO()
    if mask is not None:
        tile.paste(mask, mask=mask.split()[-1])
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] [slide-directory]')
    parser.add_option(
        '--data_path',
        dest='SLIDE_DIR',
        help='input slides',
    )
    parser.add_option(
        '--masks',
        dest='MASKS_DIR',
        default=None,
        help='input masks',
    )
    parser.add_option(
        '--model',
        default='brca',
        dest='DEEPZOOM_MODEL',
        help='input model path',
    )
    parser.add_option(
        '-B',
        '--ignore-bounds',
        dest='DEEPZOOM_LIMIT_BOUNDS',
        default=True,
        action='store_false',
        help='display entire scan area',
    )
    parser.add_option(
        '-c', '--config', metavar='FILE', dest='config', help='config file'
    )
    parser.add_option(
        '-d',
        '--debug',
        dest='DEBUG',
        action='store_true',
        help='run in debugging mode (insecure)',
    )
    parser.add_option(
        '-e',
        '--overlap',
        metavar='PIXELS',
        dest='DEEPZOOM_OVERLAP',
        type='int',
        help='overlap of adjacent tiles [1]',
    )
    parser.add_option(
        '-f',
        '--format',
        metavar='{jpeg|png}',
        dest='DEEPZOOM_FORMAT',
        help='image format for tiles [jpeg]',
    )
    parser.add_option(
        '-l',
        '--listen',
        metavar='ADDRESS',
        dest='host',
        default='127.0.0.1',
        help='address to listen on [127.0.0.1]',
    )
    parser.add_option(
        '-p',
        '--port',
        metavar='PORT',
        dest='port',
        type='int',
        default=5000,
        help='port to listen on [5000]',
    )
    parser.add_option(
        '-Q',
        '--quality',
        metavar='QUALITY',
        dest='DEEPZOOM_TILE_QUALITY',
        type='int',
        help='JPEG compression quality [75]',
    )
    parser.add_option(
        '-s',
        '--size',
        metavar='PIXELS',
        dest='DEEPZOOM_TILE_SIZE',
        type='int',
        help='tile size [254]',
    )

    (opts, args) = parser.parse_args()
    # Load config file if specified
    if opts.config is not None:
        app.config.from_pyfile(opts.config)
    # Overwrite only those settings specified on the command line
    for k in dir(opts):
        if not k.startswith('_') and getattr(opts, k) is None:
            delattr(opts, k)
    app.config.from_object(opts)
    # Set slide directory
    try:
        app.config['SLIDE_DIR'] = args[0]
    except IndexError:
        pass

    app.run(host=opts.host, port=opts.port, threaded=True)
