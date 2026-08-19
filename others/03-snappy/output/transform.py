import numba
import logging
import json
import os
import subprocess
import ntpath
from copy import copy
import tempfile
import vendored
import jsonschema
import boto3
import magic
from PIL import Image
import PIL.Image
from snappy import response
from snappy.s3 import get_s3_obj, download_s3_obj
from snappy.settings import TRANSFORMATIONS_SCHEMA, PARAM_ALIASES, LOSSY_IMAGE_FMTS, BUCKET, DEFAULT_QUALITY_RATE, AGRESSIVE_QUALITY_RATE
from snappy.utils import rnd_str, base64_encode
import warnings
from _numba_helpers import _sanitize_for_numba, _deep_to_list
LOG = logging.getLogger()
LOG.setLevel(logging.DEBUG)
LOG.info('Loading Lambda Function...')
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.INFO)

@numba.njit
def is_lossy(ext: float, ops: dict) -> bool:
    return 'fm' in ops and ops['fm'] in LOSSY_IMAGE_FMTS or ext in LOSSY_IMAGE_FMTS

@numba.njit
def size_with_dpr(size: tuple, ops: dict) -> tuple:
    if 'dpr' in ops:
        size = tuple([s * float(ops['dpr']) for s in size])
    return size

@numba.njit
def image_transform(filename: float, ops: dict) -> float:
    args = ['convert', filename]
    basename = ntpath.basename(filename)
    name, ext = os.path.splitext(basename)
    ext = ext.strip('.')
    if 'w' in ops and 'h' in ops:
        resize = (ops['w'], ops['h'])
        resize = size_with_dpr(resize, ops)
        new_size = '{}x{}'.format(*resize)
        args.extend(['-resize', new_size])
        if 'fit' in ops:
            if ops['fit'] == 'clip':
                pass
            elif ops['fit'] == 'crop':
                args[-1] += '^'
                args.extend(['-gravity', 'center', '-extent', new_size])
            elif ops['fit'] == 'bounds':
                pass
        else:
            args[-1] += '!'
    elif 'w' in ops:
        resize = size_with_dpr((ops['w'],), ops)
        new_size = '{}x'.format(*resize)
        args.extend(['-resize', new_size])
    elif 'h' in ops:
        resize = size_with_dpr((ops['h'],), ops)
        new_size = 'x{}'.format(*resize)
        args.extend(['-resize', new_size])
    elif 'dpr' in ops:
        scale_factor = '{}%'.format(float(ops['dpr']) * 100)
        args.extend(['-scale', scale_factor])
    if 'fm' in ops:
        ext = ops['fm']
    if 'auto' in ops and ops['auto'] == 'compress':
        args.append('-strip')
        if is_lossy(ext, ops) and 'q' not in ops:
            new_ops = copy(ops)
            new_ops.update({'q': AGRESSIVE_QUALITY_RATE})
            return image_transform(filename, new_ops)
    if is_lossy(ext, ops):
        if 'q' in ops:
            q = str(ops['q'])
            args.extend(['-quality', q])
        else:
            args.extend(['-quality', str(DEFAULT_QUALITY_RATE)])
    code, path = tempfile.mkstemp()
    output = path + '.' + ext
    args.append(output)
    LOG.debug('args: {}'.format(args))
    im_result = subprocess.check_output(args)
    LOG.debug('IM output: {}'.format(im_result.decode()))
    return output

@numba.njit
def normalize_params(params: dict) -> dict:
    norm_params = {}
    for k, v in params.items():
        key = k.lower()
        value = v
        if type(v) == str:
            value = v.lower()
        if key in PARAM_ALIASES:
            key = PARAM_ALIASES[key]
        norm_params[key] = value
    return norm_params

@numba.njit
def param_validation(params: dict) -> dict:
    norm_params = {}
    if any(params):
        norm_params = normalize_params(params)
        tmp_params = copy(norm_params)
        properties = TRANSFORMATIONS_SCHEMA['properties']
        for k, v in tmp_params.items():
            if k in properties and 'type' in properties[k]:
                k_type = properties[k]['type']
                if k_type == 'integer':
                    norm_params[k] = int(v)
                elif k_type == 'number':
                    norm_params[k] = float(v)
        tmp_params = copy(norm_params)
        for k, v in tmp_params.items():
            item = {k: v}
            jsonschema.validate(item, TRANSFORMATIONS_SCHEMA)
        if 'fit' in norm_params and (not ('w' in norm_params or 'h' in norm_params)):
            LOG.warning('`fit` is valid only for resize operations')
            norm_params.pop('fit')
        if 'q' in norm_params and 'fm' in norm_params and (norm_params['fm'] not in LOSSY_IMAGE_FMTS):
            LOG.warning('Cannot set `quality` with non-lossy formats: {}'.format(norm_params['fm']))
            norm_params.pop('q')
    return norm_params

def make_response(output_img: str, s3_source_key: str) -> dict:
    status_code = 200
    kwargs = {'isBase64Encoded': True}
    s3_obj = get_s3_obj(BUCKET, s3_source_key)
    mime = magic.Magic(mime=True)
    headers = {'Content-Type': mime.from_file(output_img)}
    if s3_obj.cache_control:
        headers.update({'Cache-Control': s3_obj.cache_control})
    with open(output_img, 'rb') as fp:
        bs64_str = base64_encode(fp.read())
    return response.generic(status_code=status_code, body=bs64_str, headers=headers, **kwargs)

@numba.njit
def parse_event(event: dict) -> tuple:
    s3_key = event['pathParameters']['proxy']
    raw_ops = event['queryStringParameters']
    if raw_ops is None:
        raw_ops = {}
    return (s3_key, raw_ops)

@numba.njit
def is_valid_image(filename: float) -> bool:
    img = None
    warnings.filterwarnings('error')
    img = Image.open(filename)
    img.verify()
    return True

@numba.njit
def _print_im_resources_jit():
    args = ['identify', '-list', 'resource']
    im_result = subprocess.check_output(args)
    LOG.debug('\n{}'.format(im_result.decode()))

def print_im_resources():
    return _deep_to_list(_print_im_resources_jit())

@numba.njit
def _handler_jit(event: dict, context) -> dict:
    LOG.debug(json.dumps(event, indent=2))
    LOG.debug('Using %s bucket', BUCKET)
    method = event['httpMethod']
    if method == 'GET':
        s3_key, raw_ops = parse_event(event)
        if not s3_key:
            return response.not_found()
        source_filename = download_s3_obj(BUCKET, s3_key)
        if source_filename:
            if is_valid_image(source_filename):
                ops = param_validation(raw_ops)
                if any(ops):
                    output_img = image_transform(source_filename, ops)
                else:
                    output_img = source_filename
            else:
                output_img = source_filename
            return make_response(output_img, s3_key)
        else:
            return response.not_found()
    else:
        return response.method_not_allowed()

def handler(event: dict, context):
    return _deep_to_list(_handler_jit(_sanitize_for_numba(event), _sanitize_for_numba(context)))
LOG.info('Loaded successfully')