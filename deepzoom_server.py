#!/usr/bin/env python
#
# deepzoom_server - Example web application for serving whole-slide images
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

from io import BytesIO
from optparse import OptionParser
import os
import re
from unicodedata import normalize

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

from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from run_patch_inference import analyze_one_patch
from utils_wsi import load_cfg, export_detections_to_image
import configs as CONFIGS
import torch
import tifffile
from torchvision.transforms import ToTensor

DEEPZOOM_SLIDE = None
DEEPZOOM_MASKS = None
DEEPZOOM_MODEL = None
DEEPZOOM_FORMAT = 'png'  # 'jpeg'
DEEPZOOM_TILE_SIZE = 256
DEEPZOOM_OVERLAP = 32
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 75
SLIDE_NAME = 'slide'


def run_tile(tile, model, dataset_configs, mpp=0.25, device=torch.device('cuda')):
    outputs = analyze_one_patch(
        ToTensor()(tile), model, dataset_configs, mpp=mpp, 
        compute_masks=True, device=device,
    )
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
app.config.from_envvar('DEEPZOOM_TILER_SETTINGS', silent=True)


@app.before_first_request
def load_slide():
    slidefile = app.config['DEEPZOOM_SLIDE']
    masksfile = app.config['DEEPZOOM_MASKS']
    modelpath = app.config['DEEPZOOM_MODEL']
    if slidefile is None:
        raise ValueError('No slide file specified')
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = {v: app.config[k] for k, v in config_map.items()}
    slide = open_slide(slidefile)
    app.slides = {SLIDE_NAME: DeepZoomGenerator(slide, **opts)}
    if masksfile is not None:
        masks = open_slide(masksfile)
        app.slides['masks'] = DeepZoomGeneratorRGBA(masks, **opts)

    app.associated_images = []
    app.slide_properties = slide.properties
    for name, image in slide.associated_images.items():
        app.associated_images.append(name)
        slug = slugify(name)
        app.slides[slug] = DeepZoomGenerator(ImageSlide(image), **opts)
#         print("===========")
#         print(app.slides[slug]._z_dimensions)
#         print("===========")
    try:
        mpp_x = slide.properties[openslide.PROPERTY_NAME_MPP_X]
        mpp_y = slide.properties[openslide.PROPERTY_NAME_MPP_Y]
        app.slide_mpp = (float(mpp_x) + float(mpp_y)) / 2
#         with open(slidefile, 'rb') as fb:
#             xx = tifffile.TiffFile(fb)
#             print(xx.pages[0].description)
    except (KeyError, ValueError):
        app.slide_mpp = 0

    # Run model only if masksfile is not given
    if masksfile is None and modelpath is not None:
        device = torch.device('cuda')
        if modelpath in CONFIGS.MODEL_PATHS:
            modelpath = CONFIGS.MODEL_PATHS[modelpath]
        model = torch.jit.load(modelpath)
        model.eval()
        model.to(device)
        app.model = model
        app.device = device

        meta_info = load_cfg('meta_info.yaml')
        app.dataset_configs = {'mpp': CONFIGS.DEFAULT_MPP, **CONFIGS.DATASETS, **meta_info}


@app.route('/')
def index():
    slide_url = url_for('dzi', slug=SLIDE_NAME)
    associated_urls = {
        name: url_for('dzi', slug=slugify(name)) for name in app.associated_images
    }
    return render_template(
        'slide-multipane.html',
        slide_url=slide_url,
        associated=associated_urls,
        properties=app.slide_properties,
        slide_mpp=app.slide_mpp,
    )


@app.route('/<slug>.dzi')
def dzi(slug):
    format = app.config['DEEPZOOM_FORMAT']
    try:
        resp = make_response(app.slides[slug].get_dzi(format))
        resp.mimetype = 'application/xml'
        return resp
    except KeyError:
        # Unknown slug
        abort(404)


@app.route('/<slug>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(slug, level, col, row, format):
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = app.slides[slug].get_tile(level, (col, row))
        if slug == SLIDE_NAME:
            if 'masks' in app.slides:
                mask = app.slides['masks'].get_tile(level, (col, row))
            else:
                if level == app.slides[slug]._dz_levels-1:
                    mask = run_tile(tile, app.model, app.dataset_configs, mpp=None, device=app.device)
                    mask = Image.fromarray(mask)
                else:
                    mask = None
        else:
            mask = None
    except KeyError:
        # Unknown slug
        abort(404)
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = BytesIO()
    
    # print((tile.size, tile.mode), (mask.size, mask.mode) if mask is not None else None)
    if mask is not None:
        tile.paste(mask, mask=mask.split()[-1])
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


def slugify(text):
    text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]+', '-', text)


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] [slide]')
    parser.add_option(
        '--data_path',
        dest='DEEPZOOM_SLIDE',
        help='input slides',
    )
    parser.add_option(
        '--masks',
        dest='DEEPZOOM_MASKS',
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
    
    if app.config['DEEPZOOM_SLIDE'] is None:
        parser.error('No slide file specified')

    app.run(host=opts.host, port=opts.port, threaded=True)
