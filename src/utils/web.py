import hashlib
import os
import tempfile
import zipfile
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from src.arguments.env_args import CACHE_DIR


def is_valid_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def generate_hash(url):
    return hashlib.sha256(url.encode('utf-8')).hexdigest()


def download_and_unzip(url, force_download=False, extract_to=CACHE_DIR) -> str:
    if not is_valid_url(url):
        return url
    hash_name = generate_hash(url)
    path_to_extract = os.path.join(extract_to, hash_name)

    if not force_download and os.path.exists(path_to_extract):
        return os.path.join(path_to_extract, os.listdir(path_to_extract)[0])

    os.makedirs(path_to_extract, exist_ok=True)

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('Content-Length', 0))

    progress = tqdm(response.iter_content(1024), f'Downloading {url}', total=file_size, unit='B', unit_scale=True,
                    unit_divisor=1024)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        for chunk in progress.iterable:
            tmp_file.write(chunk)
            progress.update(len(chunk))

    with zipfile.ZipFile(tmp_file.name, 'r') as thezip:
        for zip_info in thezip.infolist():
            if zip_info.filename[-1] == '/':
                continue  # skip directories
            zip_info.filename = os.path.basename(zip_info.filename)  # strip the path
            thezip.extract(zip_info, path_to_extract)

    os.unlink(tmp_file.name)  # remove the temporary file
    path_to_extract = os.path.join(path_to_extract, os.listdir(path_to_extract)[0])
    print(f"Downloaded and extracted '{url}' to '{os.path.abspath(path_to_extract)}'")

    return path_to_extract
