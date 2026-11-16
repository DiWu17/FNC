"""Read at most 64 KiB from public model/dependency hosts for troubleshooting."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from urllib.parse import urlparse

import requests

URLS = [
    "https://download.pytorch.org/whl/cu118/torch-2.7.1%2Bcu118-cp311-cp311-win_amd64.whl",
    "https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct/resolve/master/model-00001-of-00004.safetensors",
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/torch-2.7.1%2Bcu118-cp311-cp311-win_amd64.whl",
    "https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.7.1%2Bcu118-cp311-cp311-win_amd64.whl",
    "https://mirrors.nju.edu.cn/pytorch/whl/cu118/torch-2.7.1%2Bcu118-cp311-cp311-win_amd64.whl",
]


def probe(url):
    started = perf_counter()
    try:
        with requests.get(url, headers={"Range": "bytes=0-1048575"}, stream=True, timeout=(15, 25)) as response:
            first = next(response.iter_content(65536))
            return {"host": urlparse(url).hostname, "status": response.status_code,
                    "range": response.headers.get("Content-Range"), "first_bytes": len(first),
                    "seconds": round(perf_counter() - started, 2)}
    except Exception as exc:
        return {"host": urlparse(url).hostname, "error_type": type(exc).__name__,
                "seconds": round(perf_counter() - started, 2)}


if __name__ == "__main__":
    print(list(ThreadPoolExecutor(5).map(probe, URLS)))
