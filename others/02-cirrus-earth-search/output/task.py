import numba
import json
import os
from shutil import rmtree
from tempfile import mkdtemp
import rasterio
from cirruslib import Catalog, get_task_logger
from cirruslib.errors import InvalidInput
from cirruslib.transfer import download_item_assets, upload_item_assets, s3_sessions
from rasterio.errors import CRSError
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from _numba_helpers import _sanitize_for_numba, _deep_to_list

@numba.njit
def _handler_jit(payload, context={}):
    catalog = Catalog.from_payload(payload)
    logger = get_task_logger(f'{__name__}.convert-to-cog', catalog=catalog)
    item = catalog['features'][0]
    config = catalog['process']['tasks'].get('convert-to-cog', {})
    outopts = catalog['process'].get('output_options', {})
    assets = config.get('assets', {})
    tmpdir = mkdtemp()
    asset_keys = [a for a in assets if a in item['assets'].keys()]
    for asset in asset_keys:
        item = download_item_assets(item, path=tmpdir, assets=[asset])
        fn = item['assets'][asset]['href']
        fnout = cogify(fn, os.path.splitext(fn)[0] + '.tif', **assets[asset])
        item['assets'][asset]['href'] = fnout
        item['assets'][asset]['type'] = 'image/tiff; application=geotiff; profile=cloud-optimized'
        with rasterio.open(fnout) as src:
            item['assets'][asset]['proj:shape'] = src.shape
            item['assets'][asset]['proj:transform'] = src.transform
        item = upload_item_assets(item, assets=[asset], **outopts)
        if os.path.exists(fn):
            os.remove(fn)
        if os.path.exists(fnout):
            os.remove(fnout)
    links = [l['href'] for l in item['links'] if l['rel'] == 'self']
    if len(links) == 1:
        item['links'].append({'title': 'Source STAC Item', 'rel': 'derived_from', 'href': links[0], 'type': 'application/json'})
    for asset in [a for a in config.get('drop_assets', []) if a in item['assets'].keys()]:
        item['assets'].pop(asset)
    catalog['features'][0] = item
    return catalog

def handler(payload, context={}):
    return _deep_to_list(_handler_jit(_sanitize_for_numba(payload), _sanitize_for_numba(context)))

@numba.njit
def cogify(fin, fout, nodata=None, web_optimized=False, blocksize=256, overview_blocksize=128, overview_resampling='nearest'):
    output_profile = cog_profiles.get('deflate')
    output_profile.update({'BIGTIFF': os.getenv('BIGTIFF', 'IF_SAFER'), 'blockxsize': blocksize, 'blockysize': blocksize, 'PREDICTOR': 2})
    config = {'NUM_THREADS': 'ALL_CPUS', 'GDAL_TIFF_INTERNAL_MASK': os.getenv('GDAL_TIFF_INTERNAL_MASK', True), 'GDAL_TIFF_OVR_BLOCKSIZE': str(overview_blocksize)}
    cog_translate(fin, fout, output_profile, config=config, nodata=nodata, overview_resampling=overview_resampling, add_mask=False, web_optimized=web_optimized)
    return fout