"""Explicit, bounded source download of three hash-verified validation examples.

No download happens at import or app startup. Credentials and signed export URLs
must never appear in returned errors. Images stay in the local demo cache.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import threading
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import zipfile

from PIL import Image
from api_key import load_api_key
from demo_assets import ROOT, _context, inside, load_demo_assets, require

_LOCK = threading.Lock()


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, budget=24 * 1024 * 1024):
        self.url, self.size, self.budget = url, size, budget
        self.position = self.bytes_read = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        require(whence in (0, 1, 2), "Invalid seek")
        target = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        require(0 <= target <= self.size, "Invalid seek")
        self.position = target
        return target

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if not count:
            return b""
        require(count <= 8 * 1024 * 1024 and self.bytes_read + count <= self.budget,
                "Example download exceeded its size limit")
        end = self.position + count - 1
        request = Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        with urlopen(request, timeout=40) as response:
            require(response.status == 206 and response.headers.get("Content-Range") ==
                    f"bytes {self.position}-{end}/{self.size}", "Server refused a partial download")
            data = response.read(count + 1)
        require(len(data) == count, "Incomplete partial download")
        self.position += count
        self.bytes_read += count
        return data


def download_examples(root=ROOT, assets_dir="assets/demo"):
    """Return a verified bundle; sanitize all network/credential-bearing errors."""
    with _LOCK:
        try:
            return load_demo_assets(root, assets_dir)
        except FileNotFoundError:
            pass
        key = load_api_key(required=False)
        if not key:
            raise ValueError("Set your Roboflow key in .env using the assignment setup, then try again. You can also upload your own image.")
        try:
            metadata, _ = _context(root)
            dataset = metadata["dataset"]
            endpoint = (f"https://api.roboflow.com/{dataset['workspace']}/{dataset['project']}/"
                        f"{dataset['version']}/yolov8?" + urlencode({"api_key": key}))
            with urlopen(endpoint, timeout=40) as response:
                payload = json.loads(response.read(1024 * 1024))
            link = payload["export"]["link"]
            require(urlsplit(link).scheme == "https", "Export must use HTTPS")
            with urlopen(Request(link, method="HEAD"), timeout=40) as response:
                size = int(response.headers["Content-Length"])
            remote = RemoteZip(link, size)
            images = []
            with zipfile.ZipFile(remote) as archive:
                for example in metadata["examples"]:
                    name = example["source_image"]
                    matches = [member for member in archive.infolist()
                               if member.filename == name or member.filename.endswith('/' + name)]
                    require(len(matches) == 1, "Example is not uniquely present in source")
                    member = matches[0]
                    require(member.file_size <= 4 * 1024 * 1024, "Example exceeds image size limit")
                    data = archive.read(member)
                    require(hashlib.sha256(data).hexdigest() == example["image_sha256"], "Source example changed")
                    with Image.open(io.BytesIO(data)) as image:
                        require(image.size == (example["width"], example["height"]), "Source dimensions changed")
                        image.verify()
                    images.append((example["filename"], data))
            # Validate all three before writing any; never extract archive paths.
            destination = inside(root, assets_dir)
            destination.mkdir(parents=True, exist_ok=True)
            for filename, data in images:
                inside(destination, filename).write_bytes(data)
            metadata["packaged_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["download_bytes"] = remote.bytes_read
            (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            return load_demo_assets(root, assets_dir)
        except Exception:
            raise ValueError("Could not retrieve verified examples from Roboflow. Check your key and internet connection, or upload an image. No model or results were changed.") from None
