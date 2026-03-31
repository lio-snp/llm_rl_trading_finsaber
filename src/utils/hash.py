from __future__ import annotations

import hashlib
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    retries = 3
    for i in range(retries):
        h = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError as exc:
            is_timeout = isinstance(exc, TimeoutError) or getattr(exc, "errno", None) in {60, 110}
            if not is_timeout:
                raise
            if i < retries - 1:
                time.sleep(0.2 * (i + 1))
                continue
            # Keep run aggregation alive if APFS/iCloud transiently times out while reading tiny artifact files.
            try:
                st = path.stat()
                return f"timeout:{path.name}:{st.st_size}:{st.st_mtime_ns}"
            except OSError:
                return f"timeout:{path.name}"
