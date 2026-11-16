"""Resume range downloads of the official Pascal-compatible Windows wheel.

Used when a single 2.8GB HTTP response is slow. The completed wheel is verified
against the SHA256 published in PyTorch's own package index before installation.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import html
from pathlib import Path
import re
import time
from urllib.parse import unquote, urljoin, urlparse

import requests

INDEX = "https://download.pytorch.org/whl/cu118/torch/"
NAME = "torch-2.7.1+cu118-cp311-cp311-win_amd64.whl"
ROOT = Path(__file__).resolve().parents[1] / ".cache" / "wheels"
CHUNK = 64 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", choices=["official", "aliyun", "nju"], default="official")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    listing = requests.get(INDEX, timeout=30)
    listing.raise_for_status()
    links = re.findall(r'href="([^"]+)"', listing.text)
    link = next(html.unescape(value) for value in links if unquote(value.split("#")[0]).endswith(NAME))
    url = urljoin(INDEX, link.split("#")[0])
    digest = link.split("#sha256=")[1]
    if args.mirror != "official":
        base = {"aliyun": "https://mirrors.aliyun.com/pytorch-wheels/cu118/",
                "nju": "https://mirrors.nju.edu.cn/pytorch/whl/cu118/"}[args.mirror]
        url = base + NAME.replace("+", "%2B")
    destination = ROOT / NAME
    if destination.exists() and hashlib.file_digest(destination.open("rb"), "sha256").hexdigest() == digest:
        print(destination, flush=True)
        return
    with requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    part_dir = ROOT / (NAME + ".parts")
    part_dir.mkdir(exist_ok=True)

    def fetch(index):
        start, end = index * CHUNK, min((index + 1) * CHUNK, total) - 1
        part = part_dir / f"{index:04d}.part"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return index
        temporary = part.with_suffix(".tmp")
        for attempt in range(5):
            try:
                existing = temporary.stat().st_size if temporary.exists() else 0
                if existing > expected:
                    raise RuntimeError("partial range is larger than expected")
                if existing == expected:
                    temporary.replace(part)
                    return index
                resumed_start = start + existing
                with requests.get(url, headers={"Range": f"bytes={resumed_start}-{end}"}, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {resumed_start}-{end}/{total}":
                        raise RuntimeError("server did not honor requested byte range")
                    with temporary.open("ab") as output:
                        for data in response.iter_content(1024 * 1024):
                            output.write(data)
                if temporary.stat().st_size != expected:
                    raise RuntimeError("incomplete range")
                temporary.replace(part)
                return index
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)

    count = (total + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, index) for index in range(count)]
        for finished, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"Official PyTorch wheel: {finished}/{count} ranges complete", flush=True)
    temporary = destination.with_suffix(".assembling")
    actual = hashlib.sha256()
    with temporary.open("wb") as output:
        for index in range(count):
            with (part_dir / f"{index:04d}.part").open("rb") as source:
                while data := source.read(8 * 1024 * 1024):
                    actual.update(data)
                    output.write(data)
    if actual.hexdigest() != digest:
        raise RuntimeError("Official PyTorch SHA256 mismatch; wheel was not installed")
    temporary.replace(destination)
    print(f"Verified {digest}: {destination}", flush=True)


if __name__ == "__main__":
    main()
