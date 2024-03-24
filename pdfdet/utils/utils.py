import os
import fitz
from pathlib import Path
import torch
import subprocess
import cv2
import numpy as np


def pdf2png(pdf_path, save_dir):
    pdfDoc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        pix = page.get_pixmap(alpha=False)
        if not save_dir.lstrip(pdf_name):
            save_path = os.path.join(save_dir, pdf_name, f"{pg}.png")
        else:
            save_path = os.path.join(save_dir, f"{pg}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 默认是720*x尺寸
        pix._writeIMG(save_path, "png", 100)


def compute_color_for_labels(label):
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def curl_download(url, filename, *, silent: bool = False) -> bool:
    """Download a file from a url to a filename using curl."""
    silent_option = "sS" if silent else ""  # silent
    proc = subprocess.run(
        [
            "curl",
            "-#",
            f"-{silent_option}L",
            url,
            "--output",
            filename,
            "--retry",
            "9",
            "-C",
            "-",
        ]
    )
    return proc.returncode == 0


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    """
    Downloads a file from a URL (or alternate URL) to a specified path if file is above a minimum size.

    Removes incomplete downloads.
    """

    file = Path(file)
    if file.exists():
        return 0
    file.parent.mkdir(parents=True, exist_ok=True)
    assert_msg = (
        f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    )
    try:  # url1
        print(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        print(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        # curl download, retry and resume on fail
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)
