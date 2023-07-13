from __future__ import annotations

import os
import cv2
import time
import torch
import asyncio
import numbers
import databases
import sqlalchemy
import typing as t
import numpy as np
import pandas as pd
import functools
import logging
import concurrent.futures

from io import BytesIO
from PIL import Image
from threading import Lock
from collections import OrderedDict, defaultdict
from torchvision.transforms import ToTensor

from fastapi import FastAPI, Depends, BackgroundTasks, Request, Response, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseSettings
from sqlalchemy.ext.asyncio import create_async_engine
# from sqlalchemy import create_engine

import configs as hd_cfg
from utils.utils_wsi import ObjectIterator, WholeSlideDataset
from utils.utils_wsi import get_slide_and_ann_file, generate_roi_masks
from utils.utils_wsi import load_cfg, load_hdyolo_model, batch_inference
from utils.utils_wsi import export_detections_to_image, export_detections_to_table, wsi_imwrite
from utils.utils_image import Slide
from utils.deepzoom import DeepZoomGenerator


if t.TYPE_CHECKING:
    from numpy.typing import NDArray

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


db_name = "./test_nuclei.db"
database_url = f"sqlite+aiosqlite:///{db_name}"
# DATABASE_URL = "postgresql://user:password@postgresserver/db"
database = databases.Database(database_url)
metadata = sqlalchemy.MetaData()
dialect = sqlalchemy.dialects.sqlite.dialect()
db_engine = create_async_engine(
    database_url, connect_args={"check_same_thread": False, "timeout": 30},
)


class Settings(BaseSettings):
    slide_dir: str = "./slides_folder"
    masks_dir: str = "."
    slide_cache_size: int = 10

    deepzoom_format: str = "jpeg"
    deepzoom_tile_size: int = 256
    deepzoom_overlap: int = 32
    deepzoom_limit_bounds: bool = True
    deepzoom_tile_quality: int = 75
    mask_opacity: float = hd_cfg.MASK_ALPHA

    inference_tile_size: int = hd_cfg.PATCH_SIZE
    inference_overlap: int = hd_cfg.PADDING
    batch_size: int = 32
    max_latency: float = 1
    n_thread: int = 32


settings = Settings()
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class Runner(object):
    def __init__(self,
                 name: str,
                 model: str | torch.ScriptModule, 
                 device: str | torch.device = 'cpu', 
                 input_size: int = 640,
                ):
        self.torch_version = torch.__version__
        
        self.name = name
        if isinstance(model, str):
            model = load_hdyolo_model(model, nms_params=hd_cfg.NMS_PARAMS)
            print(model.headers.det.nms_params)
        elif isinstance(model, bentoml.Model):
            model = model.info.imported_module.load_model(model)
        self.model: torch.ScriptModule = model
        self.model.eval()

        if isinstance(input_size, numbers.Number):
            self.input_size = (input_size, input_size)
        else:
            self.input_size = input_size 
        
        self.to(device)
    
    def to(self, device: str | torch.device = 'cpu'):
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device: torch.device = torch.device(device)
        self.device_id: str = get_device_id(self.device)

        if self.device.type == 'cpu':  # half precision only supported on CUDA
            self.model.float()
        self.model.to(self.device)

        self.input_dtype = next(self.model.parameters()).dtype
        # warm up the runner
        dummy_inputs = (
            torch.zeros((1, 3,) + self.input_size),
            torch.ones((1, 6))
        )
        self.__call__(*dummy_inputs, compute_masks=True)

        return self
    
    def __call__(self, images: torch.Tensor, patch_infos:torch.Tensor, compute_masks=True) -> t.Dict[str, torch.Tensor]:
        images = images.to(self.device, self.input_dtype, non_blocking=True)
        with torch.no_grad():
            outputs = batch_inference(
                self.model, images, patch_infos, 
                self.input_size, compute_masks=compute_masks, 
                score_threshold=0., iou_threshold=1.
            )
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]

        return outputs


class Service(object):
    def __init__(self, n_thread=16, n_process=2, 
                 batch_size=32, buffer_size=32,
                 max_latency=1, max_halt=10, export='./tmp'):
        ## This is the queue for running inference job
        self.queue = asyncio.Queue()
        ## The buffer used to store latest visit tiles when stopped
        self.buffer = asyncio.Queue(maxsize=buffer_size)

        self.batch_size = batch_size
        self.max_latency = max_latency
        self.max_halt = 10  # automatically stop service if no data is coming in
        # self.process_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_process)
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_thread)
        self.task_manager = {}
        self._cache = {}
        self.export = export
        self.name = None
        self.status = 'stop'

    async def init_service(self, name, runner, dataset, restart=False):
        assert name is not None
        assert runner is not None
        assert dataset is not None
        await self.stop()

        self.name = name
        self.runner = runner
        self.dataset = dataset
        if self.name in self.task_manager and restart == False:
            self._cache = await self.load_cache(self.name)
        else:  # start a new service
            self.task_manager[self.name] = {
                image_info['tile_key']: idx 
                for idx, image_info in enumerate(self.dataset.images)
            }
            self._cache = {}
        self.status = 'ready'

    async def shutdown(self):
        await self.stop()
        # self.process_executor.shutdown()
        self.thread_executor.shutdown()

    async def put(self, tile_key):
        if self.status == 'run':
            if tile_key in self.task_manager[self.name]:
                idx = self.task_manager[self.name].pop(tile_key)
                await self.queue.put((tile_key, idx))
                return {'status': 200}
            else:
                return {'status': 404}
        elif self.status == 'ready':
            return self.put_buffer(tile_key)
        else:
            return {'status': self.status}
    
    def put_buffer(self, tile_key):
        if tile_key not in self._cache:
            # we don't pop from task_manager, as job may never start,
            # we push the buffer later once status change to run
            if self.buffer.full():
                try:
                    self.buffer.get_nowait()
                except:
                    pass
            try:
                self.buffer.put_nowait(tile_key)
            except:
                pass
            return {'status': 203}
        else:
            return {'status': 404}

    async def load_cache(self, name):
        table = metadata.tables[name]
        cache = await database.fetch_all(query=table.select())

        return cache

    async def start(self):
        assert self.status == 'ready', f"Run init_service first before start service."
        self.status = 'run'
        loop = asyncio.get_running_loop()

        st = time.time()
#         ## push all buffer items into queue
#         while not self.buffer.empty():
#             tile_key = await self.buffer.get()
#             await self.put(tile_key)

        try:
            idle_st = time.time()
            while time.time() - idle_st < self.max_halt:
                # fill up the queue with random seed when in low demand
                if self.status == 'run':
                    st = time.time()
                    while not self.buffer.empty() and self.queue.qsize() < 16 and len(self.task_manager[self.name]):
                        tile_key = await self.buffer.get()
                        await self.put(tile_key)

                    if self.queue.qsize() < 2 and len(self.task_manager[self.name]):
                        tile_key, idx = self.task_manager[self.name].popitem()
                        await self.queue.put((tile_key, idx))

                images, patch_infos, tile_keys = [], [], []
                while len(images) < self.batch_size and time.time() - idle_st < self.max_latency:
                    tile_key, idx = await self.queue.get()
                    # spread __getitem__ to threads for cpu busy preprocessing steps
                    image, patch_info = await loop.run_in_executor(
                        self.thread_executor, self.dataset.__getitem__, idx
                    )  # image, patch_info = self.dataset[idx]
                    images.append(image)
                    tile_keys.append(tile_key)
                    patch_infos.append(patch_info)

                if images:
                    images, patch_infos = torch.stack(images), torch.stack(patch_infos)
                    asyncio.run_coroutine_threadsafe(self.batch_run(images, patch_infos, tile_keys), loop)
                    idle_st = time.time()  # reset timer

                # put neighbors into buffer
                st = time.time()
                offsets = np.array([[0,-1,-1],[0, -1, 0],[0,-1,1],[0,0,-1],[0,0,1],[0,1,-1],[0,1,0],[0,1,1]])
                neighbors = np.array(tile_keys)
                neighbors = np.concatenate([neighbors + offset for offset in offsets])
                new_keys = set(tuple(_) for _ in neighbors) - set(tile_keys)
                for tile_key in new_keys:
                    self.put_buffer(tile_key)

            self.status = 'ready'

        except KeyboardInterrupt:
            raise 'KeyboardInterrupt'

    async def stop(self):
        if self.status == 'run':
            self.status = 'ready'
            try:
                # finish elements in queue.
                while not self.queue.empty():
                    logging.info(f"Finishing {self.queue.qsize()} tasks remaining in queue.")
                    await asyncio.sleep(2)
            except KeyboardInterrupt:
                raise 'KeyboardInterrupt'
            logging.info("Service stopped", await self.monitor())

    async def kill(self):
        await self.stop()
        
        if self._cache:
            self.export_cache_to_db(self._cache.keys())
        self.status = 'stop'

    async def batch_run(self, images, patch_infos, tile_keys):
        outputs = self.runner(images, patch_infos, compute_masks=True,)
        for tile_key, output in zip(tile_keys, outputs):
            self._cache[tile_key] = output

        return {'status': 200}
    
    async def export_cache_to_db(self, tile_keys):
        results, tile_infos = defaultdict(list), []
        for tile_key in tile_keys:
            for k, v in self._cache[tile_key].items():
                results[k].append(v)
                tile_infos += ['{}_{}_{}'.format(*tile_key)] * len(v['boxes'])          
        results = {k: torch.cat(v) for k, v in results.items()}
        df = export_detections_to_table(results, save_masks=True)
        df['tile_key'] = tile_infos

        await to_sql(df, self.name, db_engine, if_exists='append', index=False, method='multi')
    
    async def fetch_from_table(self, tile_key: str):
        table = metadata.tables[self.name]
        query = table.select().where(table.columns.tile_key == tile_key)
        rows = await database.fetch_all(query=query)
        df = pd.DataFrame(rows, columns=table.columns.keys()).set_index('id')

        return df

    async def fetch_and_plot_tile(self, tile_key: tuple):
        # ignore request if status != 'run' and tile_key is not precaclulated
        if self.status != 'run' and tile_key not in app.service._cache:
            return None

        # try add tile_key to queue if it's not ready
        if tile_key in self.task_manager[self.name]:
            await self.put(tile_key)
            await asyncio.sleep(2)

        idx = self.dataset.indices.get(tile_key)
        x0, y0, w_roi, h_roi = self.dataset.images[idx]['data']['coord']

        mask_img = None
        if tile_key in self._cache:  # directly fetch from cache
            res = {**app.service._cache[tile_key]}
            res['boxes'] = res['boxes'] - torch.tensor([x0, y0, x0, y0])
            if res is not None and len(res['boxes']):
                object_iterator = ObjectIterator(
                    boxes=res['boxes'], 
                    labels=res['labels'], 
                    scores=res['scores'], 
                    masks=res['masks'] if 'masks' in res else None,
                )
                mask_img = export_detections_to_image(
                    object_iterator, (h_roi, w_roi), 
                    app.dataset_configs['labels_color'],
                    save_masks=True, border=3, alpha=settings.mask_opacity)
                mask_img = Image.fromarray(mask_img)
#         else:  # fetch from db, not used for now.
#             dets = await app.service.fetch_from_table('{}_{}_{}'.format(*tile_key))
#             if len(dets):
#                 res = {
#                     'boxes': torch.from_numpy(dets[['x0', 'y0', 'x1', 'y1']].to_numpy() - np.array([x0, y0, x0, y0])),
#                     'labels': torch.from_numpy(dets['label'].to_numpy()),
#                     'masks': [[np.array([list(map(float, x.split(','))), list(map(float, y.split(',')))]).T - np.array([x0,y0])]
#                               for x, y in zip(dets['poly_x'], dets['poly_y'])], 
#                 }
#             else:
#                 res = None
#             if res is not None and len(res['boxes']):
#                 mask_img = export_detections_to_image(
#                     res, (h_roi, w_roi), 
#                     app.dataset_configs['labels_color'],
#                     save_masks=True, border=3, alpha=settings.mask_opacity)
#                 mask_img = Image.fromarray(mask_img)

        return mask_img

    async def monitor(self):
        loginfo = {
            "name": self.name,
            "status": self.status, 
            "queue_size": self.queue.qsize(),
            "buffer_size": self.buffer.qsize(),
            "percentage": 1 - len(self.task_manager[self.name])/len(self.dataset),
            # "db_size": None, 
        }
        return loginfo


def get_device_id(device):
    if device.type == 'cpu':
        return 'cpu'
    return f"{device.type}:{device.index}"


class _SlideCache:
    def __init__(self, cache_size, dz_opts, dataset_configs):
        self.cache_size = cache_size
        self.dz_opts = dz_opts
        self._lock = Lock()
        self._cache = OrderedDict()
        self.dataset_configs = dataset_configs

    # @functools.lru_cache()
    def get(self, slide_path):
        # slide_path is the full absolute path
        with self._lock:
            if slide_path in self._cache:
                # Move to end of LRU
                slide, dataset = self._cache.pop(slide_path)
                self._cache[slide_path] = (slide, dataset)
                return slide, dataset

        ## Load slide reader, inference dataset and deepzoom generator
        print(f"======================================")
        print(f"Prepare slide: {slide_path}, ", end="")
        osr = Slide(*get_slide_and_ann_file(slide_path))
        osr.attach_reader(openslide.open_slide(osr.img_file), engine='openslide')
        print(f"attach file success.")

        print(f"Build WholeSlideDataset: ", end="")
        t0 = time.time()
        roi_masks = generate_roi_masks(osr, 'tissue')
        dataset = WholeSlideDataset(osr, masks=roi_masks, processor=None, **self.dataset_configs)
        print(f"{time.time()-t0} s\n{dataset}")
        print(f"Build DeepZoomGenerator: ", end="")
        t0 = time.time()
        slide = DeepZoomGenerator(osr, **self.dz_opts)
        slide.filename = os.path.basename(slide_path)
        print(f"{time.time() - t0} s\n{slide}")
        print(f"======================================")

        with self._lock:
            if slide_path not in self._cache:
                if len(self._cache) == self.cache_size:
                    k, (s, d) = self._cache.popitem(last=False)
                    d.slide.detach_reader(close=True)
                self._cache[slide_path] = (slide, dataset)

        return slide, dataset


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
            # elif OpenSlide.detect_format(cur_path):
            elif cur_path.endswith('.svs'):
                self.children.append(_SlideFile(cur_relpath))


class _SlideFile:
    def __init__(self, relpath):
        self.name = os.path.basename(relpath)
        self.url_path = relpath


class _Devices:
    def __init__(self):
        self.list = ['cpu'] + [f'cuda:{idx}' for idx in range(torch.cuda.device_count())]


def _get_slide(path):
    slide_path = os.path.abspath(os.path.join(app.slide_dir, path))
    if not slide_path.startswith(app.slide_dir + os.path.sep):
        logging.exception(f"{slide_path} is invalid.")
        raise HTTPException(status_code=404, detail=f"{slide_path} is invalid.")
    if not os.path.exists(slide_path):
        logging.exception(f"{slide_path} not found.")
        raise HTTPException(status_code=404, detail=f"{slide_path} not found.")
    try:
        slide, dataset = app.cache.get(slide_path)
        return slide, dataset
    except Exception as e:
        logging.exception(f"Failed to load {slide_path}.")
        raise HTTPException(status_code=404, detail=f"Failed to load {slide_path}.")


async def create_table(table_name: str):
    # Create tables if not exists
    if table_name not in metadata.tables:
        table = sqlalchemy.Table(
            table_name,
            metadata,
            sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column("tile_key", sqlalchemy.UnicodeText),
            sqlalchemy.Column("x0", sqlalchemy.Float),
            sqlalchemy.Column("y0", sqlalchemy.Float),
            sqlalchemy.Column("x1", sqlalchemy.Float),
            sqlalchemy.Column("y1", sqlalchemy.Float),
            sqlalchemy.Column("label", sqlalchemy.Integer),
            sqlalchemy.Column("score", sqlalchemy.Float),
            # we add masks later. 
            sqlalchemy.Column("poly_x", sqlalchemy.UnicodeText, nullable=True),
            sqlalchemy.Column("poly_y", sqlalchemy.UnicodeText, nullable=True),
        )
        schema = sqlalchemy.schema.CreateTable(table, if_not_exists=True)
        query = str(schema.compile(dialect=dialect))
        await database.execute(query=query)

    return table_name


def get_service_name(path: str, model: str):
    return f"{path}_{model}"


@app.on_event("startup")
async def _setup():
    app.slide_dir = os.path.abspath(settings.slide_dir)
    app.masks_dir = os.path.abspath(settings.masks_dir)
    opts = {
        'tile_size': settings.deepzoom_tile_size,
        'overlap': settings.deepzoom_overlap,
        'limit_bounds': settings.deepzoom_limit_bounds,
    }

    meta_info = load_cfg('meta_info.yaml')
    dataset_configs = {'mpp': hd_cfg.DEFAULT_MPP, **hd_cfg.DATASETS, **meta_info}
    print(f"==========startup==========")
    print(opts, dataset_configs)
    print(f"===========================")
    app.cache = _SlideCache(settings.slide_cache_size, opts, dataset_configs)
    app.dataset_configs = dataset_configs
    app.runner = {'name': '', 'runner': None}
    app.device = 'cpu'
    app.slide_path = None

    await database.connect()  # connect to database
    app.service = Service(
        n_thread=settings.n_thread,
        n_process=2, 
        batch_size=settings.batch_size, 
        buffer_size=32, 
        max_latency=settings.max_latency,
        max_halt=10, 
        export=db_name,
    )


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    await app.service.shutdown()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    await app.service.kill()

    context = {
        "request": request, 
        "root_dir": _Directory(app.slide_dir),
        "models": hd_cfg.MODEL_PATHS.keys(),
        "devices": _Devices(),
        "curr_device": app.device,
        "curr_model": app.runner['name'],
        "curr_slide": app.slide_path,
    }
    return templates.TemplateResponse('index.html', context)


@app.post("/models")
async def register_model(model: str = '', device: str = 'cpu'):
    if device != 'cpu' and torch.cuda.mem_get_info(device)[0]/1e8 > 8.:  # require 8 gb memory
        app.device = device

    if model:
        if model == app.runner['name']:
            try:
                app.runner['runner'].to(app.device)
                app.device = app.runner['runner'].device_id  # in case moved
                logging.info(f"Load runner model={model} to {app.device} success.")
            except Exception as e:
                logging.exception(f"Load runner {model} to {app.device} failed.")
        else:
            try:
                input_size = (settings.inference_tile_size + 
                              2 * settings.inference_overlap)
                runner = Runner(
                    name=model, 
                    model=hd_cfg.MODEL_PATHS[model], 
                    device=app.device, 
                    input_size=input_size,
                )
                app.runner = {'name': runner.name, 'runner': runner}
                app.device = runner.device_id
                logging.info(f"Build runner model={runner.name} on ({runner.device_id}) success.")
            except Exception as e:
                logging.exception(f"Build runner {model} failed!")

    resp = {'model': app.runner['name'], 'device': app.device}

    return resp


@app.post("/service")
async def service_control(toggle: bool, background_tasks: BackgroundTasks):
    if toggle:
        try:
            ## TODO: set restart=False when finish writing db
            assert app.slide_path and app.runner['name']
            service_name = get_service_name(app.slide_path, app.runner['name'])
            if app.service.status != 'ready' or app.service.name != service_name:
                slide, dataset = _get_slide(app.slide_path)
                await app.service.init_service(
                    name=service_name,
                    runner=app.runner['runner'], 
                    dataset=dataset, 
                    restart=True,
                )
            logging.info(f"Start service for {app.slide_path} with runner {app.runner['name']}.")
            await app.service.start()
        except Exception as e:
            logging.exception(f"Failed to start service for {app.slide_path} with runner {app.runner['name']}.")
    else:
        await app.service.stop()
    
    return {'status': app.service.status}


@app.get("/slides/{path}", response_class=HTMLResponse)
async def slide(path: str, request: Request):
    # force stop when refresh page or direct to new slide
    if path != app.slide_path:
        await app.service.kill()
    else:
        await app.service.stop()

    slide, dataset = _get_slide(path)  # path is relative path to app.slide_dir
    app.slide_path = path
    slide_url = app.url_path_for('dzi', path=path)
    masks_url = app.url_path_for('dzi_masks', path=path)

    context = {
        "request": request,
        "root_dir": _Directory(app.slide_dir),
        "models": hd_cfg.MODEL_PATHS.keys(),
        "devices": _Devices(),
        "curr_device": app.device,
        "curr_model": app.runner['name'],
        "curr_slide": app.slide_path,
        "slide_url": slide_url,
        "masks_url": masks_url,
        "slide_mpp": slide.mpp,
    }

    return templates.TemplateResponse('index.html', context)


@app.get("/slides/{path}/dzi")
def dzi(path: str):
    slide, dataset = _get_slide(path)
    dzi = slide.get_dzi(settings.deepzoom_format)
    resp = Response(content=dzi, media_type="application/xml")

    return resp


@app.get("/masks/{path}/dzi")
def dzi_masks(path: str):
    slide, dataset = _get_slide(path)
    dzi = dataset.get_dzi(format='png')
    resp = Response(content=dzi, media_type="application/xml")

    return resp


@app.get("/slides/{path}/dzi_files/{level}/{patch_file}")
async def slides_tile(path: str, level: int, patch_file: str):
    tmp, format = patch_file.split('.')
    col, row = [int(x) for x in tmp.split('_')]
    slide, dataset = _get_slide(path)
    
    if level == slide.level_count-1:
        if app.service.status != 'stop':
            st = time.time()
            factor = slide.tile_size / dataset.window_size
            # if factor < 1. + 1e-4:
            inf_col, inf_row = round(col * factor), round(row * factor)
            tasks = [app.service.put((0, inf_col+c, inf_row+r)) for c in [-1, 0, 1] for r in [-1, 0, 1]]
            responses = await asyncio.gather(*tasks)

    format = format.lower()
    if format != 'jpeg' and format != 'png':
        raise HTTPException(status_code=404, detail=f"{format} is not supported.")
    try:
        tile = slide.get_tile(level, (col, row))
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Invalid level or coordinates.")

    buf = BytesIO()
    tile.save(buf, format, quality=settings.deepzoom_tile_quality)
    resp = Response(content=buf.getvalue(), media_type=f"image/{format}")

    return resp


@app.get("/masks/{path}/dzi_files/{level}/{patch_file}")
async def masks_tile(path: str, level: int, patch_file: str, background_tasks: BackgroundTasks):
    tmp, format = patch_file.split('.')
    col, row = [int(x) for x in tmp.split('_')]
    slide, dataset = _get_slide(path)

    mask_img = None
    if level == slide.level_count-1 and app.service.status != 'stop':
        mask_img = await app.service.fetch_and_plot_tile((dataset.page, col, row))

    if mask_img is not None:
        buf = BytesIO()
        mask_img.save(buf, format)
        resp = Response(content=buf.getvalue(), media_type=f"image/{format}")

        return resp


async def to_sql(data, name, con, **kwargs):
    fn = lambda con, df, name, **kwargs: df.to_sql(name, con, **kwargs)
    async with con.begin() as connect:
        await connect.run_sync(fn, data, name, **kwargs)


async def read_sql(query, con, **kwargs):
    fn = lambda con, sql, **kwargs: pd.read_sql(sql, con, **kwargs)
    async with con.begin() as connect:
        df = await connect.run_sync(fn, query, **kwargs)
    return df


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
